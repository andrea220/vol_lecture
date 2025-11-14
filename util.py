from collections import defaultdict
import pandas as pd
from datetime import date, timedelta
import QuantLib as ql
from dateutil.relativedelta import relativedelta
import numpy as np
from scipy.interpolate import RegularGridInterpolator

class DynamicPicPac:

    def __init__(self,
                 data: pd.DataFrame,
                 vol_df: pd.DataFrame,
                 initial_capital: float,
                 investment_characteristics: pd.DataFrame,
                 initial_equity,
                 max_investment: float,
                 investment_horizon: int,
                 put_strike: float,
                 call_strike: float,
                 collar_strategy: bool,
                 drawdown_treshold: float,
                 reinvest_in_equity: bool,
                 eom: bool,
                 
                 ) -> None:
        """Phased equity allocation with dynamic short-put/long-call collar overlay.

        Collars are added alongside new equity allocations and unwound on drawdown triggers.
        """
        self.data = data.copy()
        self.vol_df = vol_df.copy()
        self.initial_capital = initial_capital
        self.investment_characteristics = investment_characteristics
        self.initial_equity = initial_equity
        self.max_investment = max_investment
        self.investment_horizon = investment_horizon
        self.collar_low = put_strike
        self.collar_up = call_strike
        self.collar_strategy = collar_strategy#
        self.drawdown_treshold = drawdown_treshold
        self.reinvest_in_equity = reinvest_in_equity
        self.eom = eom


    def get_investment_dates(self, all_dates, investment_characteristics):
        """
        Restituisce una lista di date corrispondenti a t0 + n mesi, 
        dove t0 è la prima data in all_dates.
        Se la data teorica non esiste in all_dates, ritorna la prima successiva disponibile.
        """
        all_dates = sorted(pd.to_datetime(all_dates))
        t0 = all_dates[0]
        result_dates = []
        
        for n in investment_characteristics['Period']:
            target_date = t0 + relativedelta(months=int(n))
            
            # Trova la prima data disponibile in all_dates >= target_date
            next_dates = [d for d in all_dates if d >= target_date]
            if next_dates:
                result_dates.append(next_dates[0].date())
            else:
                result_dates.append(all_dates[-1].date())  # se oltre l’ultima disponibile
        
        return result_dates
    
    def get_investment_schedule(self, all_dates, investment_characteristics):
        """
        Restituisce un dizionario {data: nominal} per ciascun periodo definito.
        """
        investment_dates = self.get_investment_dates(all_dates, investment_characteristics)
        return dict(zip(investment_dates, investment_characteristics['Nominal']))

    def on_start(self):
        return
    
    def run_single_backtest(self, start_date: date, debug: bool = False):
        period_data = self.data[
                            (self.data['date'] >= start_date) &
                            (self.data['date'] <= start_date + timedelta(365* self.investment_horizon) )
                            ].copy()

        all_dates = period_data['date'].values
        end_date = all_dates.max()
        investment_schedule = self.get_investment_schedule(all_dates, self.investment_characteristics)

        daily_data = period_data.reset_index(drop=True)
        initial_price = daily_data.iloc[0]['close']

        results = []
        portfolio_value = self.initial_capital 
        safe_part = self.initial_capital
        total_invested = 0  # Track total invested amount
        risky_part = 0  # Track current investment value

        # COLLAR
        drawdown_condition = False 
        add_unwind = False
        # unwinded_pnl = 0
        collar_pnl = 0
        positions = {}
        current_drawdown = 0

        first_of_month = first_date_per_month(all_dates)

        self.rf_t_grid = np.arange(0.1, 5.1, 0.25)
        self.rf_curves_debug = np.zeros((len(daily_data), len(self.rf_t_grid)))
        
        # Initialize debug variables for collar pricing parameters
        if debug:
            self.collar_pricing_debug = []  # List to store all collar pricing parameters
        
        for i in range(len(daily_data)):
            df_tmp = daily_data.iloc[:i+1,:]
            prev_date_period = df_tmp.iloc[i-1,0]
            current_date_period = df_tmp.iloc[i,0]
            tau = (current_date_period - prev_date_period).days/365

            prev_price = df_tmp.iloc[i - 1]['close']
            current_price = df_tmp.iloc[-1]['close']
            performance = current_price / prev_price

            vol_df_tmp = self.vol_df[self.vol_df['date'] == current_date_period].copy()
            vol_df_tmp['moneyness'] *= current_price
            if vol_df_tmp.empty:
                raise ValueError(f"No volatility data found for date: {current_date_period}")
            surface = VolHandle(vol_df_tmp)

            risk_free = df_tmp.iloc[- 1]['rate1']
            risk_free2 = df_tmp.iloc[- 1]['rate2']
            risk_free4 = df_tmp.iloc[- 1]['rate4']
            risk_free5 = df_tmp.iloc[- 1]['rate5']

            rf_times = np.array([1, 2, 4, 5])
            rf_rates = np.array([risk_free, risk_free2, risk_free4, risk_free5])
            # np.interp by default uses left and right values for extrapolation, so for t<1 it returns risk_free
            risk_free_interp = lambda t: float(np.interp(t, rf_times, rf_rates, left=risk_free, right=risk_free5))
            # if debug:
            #     self.rf_curves_debug[i, :] = np.array([risk_free_interp(t) for t in self.rf_t_grid])
                
            div_yield = df_tmp.iloc[- 1]['div_yield']


            if i == 0: # primo giorno
                new_investment = self.initial_equity
                total_invested += new_investment
                risky_part += new_investment
                safe_part -= new_investment

                # posizioni collar:
                if self.collar_strategy:
                    collar_quantity = new_investment/current_price
                    tau_expiry = (end_date-current_date_period).days/365 # time to maturity, da data di valutazione a end_date
                    rf_T = risk_free_interp(tau_expiry) 
                    vol_put = surface.get_vol(self.collar_low * current_price, tau_expiry) # vola implicita put
                    vol_call = surface.get_vol(self.collar_up * current_price, tau_expiry) # vola implicita call
                    put = option_price(current_date_period, end_date, current_price, self.collar_low*current_price, 
                                                vol_put, rf_T, div_yield, ql.Option.Put)
                    call = option_price(current_date_period, end_date, current_price, self.collar_up*current_price,
                                        vol_call, rf_T, div_yield, ql.Option.Call)
                    
                    collar_value = put - call 
                    positions[f'collar_{i}'] = {
                                                    'initial_price': collar_value,
                                                    'quantity': collar_quantity,
                                                    'trade_date': current_date_period,
                                                    'maturity': end_date,
                                                    'tau': tau_expiry,
                                                    'put_price': put,
                                                    'call_price': call,
                                                    'risk_free': rf_T,
                                                    'div_yield': div_yield,
                                                    'put_strike': self.collar_low*current_price,
                                                    'put_vol': vol_put,
                                                    'call_strike': self.collar_up*current_price,
                                                    'call_vol': vol_call,
                                                }
                    
                    # Save collar pricing parameters for debug
                    if debug:
                        self.collar_pricing_debug.append({
                            'iteration': i,
                            'date': current_date_period,
                            'event_type': 'initial_collar',
                            'spot_price': current_price,
                            'put_strike': self.collar_low*current_price,
                            'call_strike': self.collar_up*current_price,
                            'vol_put': vol_put,
                            'vol_call': vol_call,
                            'tau_expiry': tau_expiry,
                            'risk_free': rf_T,
                            'div_yield': div_yield,
                            'put_price': put,
                            'call_price': call,
                            'collar_value': collar_value,
                            'collar_quantity': collar_quantity,
                            'maturity': end_date
                        })

                # Salva risultati
                results.append({
                    'Start_Date': prev_date_period,
                    'End_Date': current_date_period,
                    'New_Investment': new_investment,
                    'Current_Investment': risky_part,
                    'Total_Invested': total_invested,
                    'Initial_Price': initial_price,
                    'Index Price': current_price,
                    'Index Return': performance - 1,
                    'Portfolio_Value': portfolio_value,
                    'Bond part': safe_part,
                    'Equity part': risky_part,
                    'Collar PnL': collar_pnl,
                    'Unwind': add_unwind
                })
                continue
            
            ######## RIVALUTAZIONI A OGNI T
            collar_pnl = 0
            collar_net_position = 0
            if self.collar_strategy and not drawdown_condition:
                prev_drawdown = current_drawdown
                current_drawdown = calculate_max_drawdown(df_tmp, start_date, current_date_period)
                for key, value in positions.items():
                    if current_date_period == end_date:
                        put = max(value['put_strike'] - current_price, 0)
                        call = max(current_price - value['call_strike'], 0)
                        final_pnl = ((put - call) - value['initial_price']) * value['quantity'] 
                        collar_pnl += final_pnl
                        continue
                    
                    tau_expiry = (end_date-current_date_period).days/365
                    rf_T = risk_free_interp(tau_expiry) 
                    vol_put = surface.get_vol(value['put_strike'], tau_expiry)
                    vol_call = surface.get_vol(value['call_strike'], tau_expiry)
                    put = option_price(current_date_period, end_date, current_price, value['put_strike'], 
                                        vol_put, rf_T, div_yield, ql.Option.Put)
                    call = option_price(current_date_period, end_date, current_price, value['call_strike'],
                                        vol_call,  rf_T, div_yield, ql.Option.Call)
                    value[f'pnl_M{i}'] = ((put - call) - value['initial_price']) * value['quantity'] 

                    collar_pnl += value[f'pnl_M{i}']
                    collar_net_position += value['quantity']
                    
                    # Save collar pricing parameters for debug
                    if debug:
                        self.collar_pricing_debug.append({
                            'iteration': i,
                            'date': current_date_period,
                            'event_type': 'revaluation',
                            'collar_key': key,
                            'spot_price': current_price,
                            'put_strike': value['put_strike'],
                            'call_strike': value['call_strike'],
                            'vol_put': vol_put,
                            'vol_call': vol_call,
                            'tau_expiry': tau_expiry,
                            'risk_free': rf_T,
                            'div_yield': div_yield,
                            'put_price': put,
                            'call_price': call,
                            'collar_value': put - call,
                            'collar_quantity': value['quantity'],
                            'pnl': value[f'pnl_M{i}'],
                            'maturity': end_date
                        })

            # se c'è stato unwind all'iterazione precedente si aggiorna la risky part:
            if add_unwind:
                # print("prev risky: ", risky_part)
                risky_part += unwinded_pnl
                # print("new risky: ", risky_part)
                add_unwind = False

            # Aggiorna parte bond (crescita con interesse risk-free a t1)
            safe_part *= (1 + (risk_free * tau))
            # Aggiorna parte equity
            risky_part *= performance
            risky_log = risky_part

            # Valore portafoglio aggiornato
            portfolio_value = safe_part + risky_part + collar_pnl

            # Trigger di drawdown dell'indice 
            # quando l'indice supera la soglia di drawdown vengono unwindati tutti i collar 
            if self.eom and not drawdown_condition:
                # if current_date_period in first_of_month:
                    # print("checking dd: ", current_date_period)

                # VECCHIO CODICE (commentato): controllava il drawdown massimo storico
                # if not drawdown_condition and self.collar_strategy and abs(current_drawdown) >= self.drawdown_treshold and current_date_period in first_of_month:
                #     print("found dd: ", current_date_period)
                #     print("*")
                #     for key, value in positions.items():
                #         value['quantity'] = 0
                #     drawdown_condition = True 
                #     unwinded_pnl =  collar_pnl
                #     add_unwind = True
                
                # NUOVO CODICE: controlla il drawdown corrente alla data first_of_month
                if not drawdown_condition and self.collar_strategy and current_date_period in first_of_month:
                    # Calcola il drawdown CORRENTE (non massimo storico) alla data first_of_month
                    running_max = df_tmp['close'].max()
                    current_price_drawdown = (current_price - running_max) / running_max
                    
                    # Triggera unwind solo se il prezzo è ancora sotto la soglia alla data first_of_month
                    if abs(current_price_drawdown) >= self.drawdown_treshold:
                        print("found dd: ", current_date_period)
                        # print("current drawdown: ", current_price_drawdown)
                        # print("drawdown daily: ", current_drawdown)
                        print("*")
                        for key, value in positions.items():
                            value['quantity'] = 0
                        drawdown_condition = True 
                        unwinded_pnl =  collar_pnl # si salva il pnl chiuso in questa variabile che al loop successivo viene aggiunta al portafoglio
                        # print("unwind pnl: ", unwinded_pnl)
                        add_unwind = True # salva il flag così verrà inserito il capitale di unwind alla prossima iterazione 
            else:
                if not drawdown_condition and self.collar_strategy and abs(current_drawdown) >= self.drawdown_treshold and abs(prev_drawdown) < self.drawdown_treshold:
                    print("found dd: ", current_date_period)
                    print("*")
                    for key, value in positions.items():
                        value['quantity'] = 0
                    drawdown_condition = True 
                    unwinded_pnl =  collar_pnl # si salva il pnl chiuso in questa variabile che al loop successivo viene aggiunta al portafoglio
                    add_unwind = True # salva il flag così verrà inserito il capitale di unwind alla prossima iterazione 

            # Aggiorna la posizione Risky alle date di investimento
            new_investment = 0
            if total_invested < self.max_investment:
                if current_date_period in investment_schedule.keys(): # se siamo in una data di investimento
                    new_investment = investment_schedule[current_date_period] * (self.max_investment - self.initial_equity)
                    total_invested += new_investment
                    risky_part += new_investment
                    safe_part -= new_investment

                    if not drawdown_condition and self.collar_strategy:
                        # posizioni collar
                        collar_quantity = new_investment/current_price
                        tau_expiry = (end_date-current_date_period).days/365 # time to maturity, da data di valutazione a end_date
                        rf_T = risk_free_interp(tau_expiry) 
                        vol_put = surface.get_vol(self.collar_low * current_price, tau_expiry) # vola implicita put
                        vol_call = surface.get_vol(self.collar_up * current_price, tau_expiry) # vola implicita call
                        put = option_price(current_date_period, end_date, current_price, self.collar_low*current_price, 
                                                    vol_put, rf_T, div_yield, ql.Option.Put)
                        call = option_price(current_date_period, end_date, current_price, self.collar_up*current_price,
                                            vol_call, rf_T, div_yield, ql.Option.Call)
                        collar_value = put - call 
                        positions[f'collar_{i}'] = {
                                                        'initial_price': collar_value,
                                                        'quantity': collar_quantity,
                                                        'trade_date': current_date_period,
                                                        'maturity': end_date,
                                                        'tau': tau_expiry,
                                                        'put_strike': self.collar_low*current_price,
                                                        'put_vol': vol_put,
                                                        'call_strike': self.collar_up*current_price,
                                                        'call_vol': vol_call,
                                                    }
                        
                        # Save collar pricing parameters for debug
                        if debug:
                            self.collar_pricing_debug.append({
                                'iteration': i,
                                'date': current_date_period,
                                'event_type': 'new_investment_collar',
                                'spot_price': current_price,
                                'put_strike': self.collar_low*current_price,
                                'call_strike': self.collar_up*current_price,
                                'vol_put': vol_put,
                                'vol_call': vol_call,
                                'tau_expiry': tau_expiry,
                                'risk_free': rf_T,
                                'div_yield': div_yield,
                                'put_price': put,
                                'call_price': call,
                                'collar_value': collar_value,
                                'collar_quantity': collar_quantity,
                                'maturity': end_date
                            })

            # Salva risultati
            results.append({
                'Start_Date': prev_date_period,
                'End_Date': current_date_period,
                'New_Investment': new_investment,
                'Current_Investment': risky_part,
                'Total_Invested': total_invested,
                'Initial_Price': initial_price,
                'Drawdown': current_drawdown,
                'Index Price': current_price,
                'Index Return': performance - 1,
                'Portfolio_Value': portfolio_value,
                'Bond part': safe_part,
                'Equity part': risky_log,
                'Collar PnL': collar_pnl,
                'Unwind': add_unwind
            })
        results = pd.DataFrame(results)
        positions = pd.DataFrame(positions)
        return results, positions
    
    def run_rolling_backtests(self, frequency= "1M", batch = 'benchmark', debug: bool = False):
        """Run rolling backtests starting each month.

        Returns a concatenated results DataFrame and a dict of positions per start date.
        """
        all_results = []
        # Initialize collar_pricing_debug to collect data from all backtests
        if debug:
            self.collar_pricing_debug = []
        
        start_dates = self.data['date'].values
        if frequency == "1M":
            start_dates = first_date_per_month(start_dates)
        else:
            start_dates = middle_date_per_month(start_dates)
        
        # i = 0
        for start_date in start_dates:
            print('## CALCULATING: ', start_date)
            # i += 1
            # if i == 2:
            #     break
            

            if start_date + timedelta(365* self.investment_horizon) > self.data['date'].max():
                break

            result, _ = self.run_single_backtest(start_date, debug)
            if result is not None:
                summary = analyze_results(result)
                all_results.append(summary)
        
        res_out = pd.concat(all_results, ignore_index=True)
        
        # Export results to Excel with multiple sheets
        output_file = f"results/{batch}_results.xlsx"
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            res_out.to_excel(writer, sheet_name='Results', index=False)
            
            # Calculate and export collar pricing debug summary if available
            if debug and hasattr(self, 'collar_pricing_debug') and self.collar_pricing_debug:
                collar_summary = analyze_collar_pricing_debug(self.collar_pricing_debug)
                collar_summary.to_excel(writer, sheet_name='Collar_Pricing_Stats', index=False)

        return res_out
    
def analyze_collar_pricing_debug(collar_pricing_debug):
    """Calculate summary statistics (mean, min, max) for collar pricing debug data.
    
    Parameters:
        collar_pricing_debug: List of dictionaries containing collar pricing parameters
        
    Returns:
        DataFrame with mean, min, max for each numeric parameter
    """
    if not collar_pricing_debug:
        return pd.DataFrame()
    
    # Convert list of dicts to DataFrame
    df = pd.DataFrame(collar_pricing_debug)
    
    # Select numeric columns to analyze
    numeric_cols = [
        'spot_price', 'put_strike', 'call_strike',
        'vol_put', 'vol_call', 'tau_expiry', 
        'risk_free', 'div_yield', 'put_price', 
        'call_price', 'collar_value', 'collar_quantity'
    ]
    
    # Filter to only columns that exist in the dataframe
    numeric_cols = [col for col in numeric_cols if col in df.columns]
    
    summary_stats = []
    for col in numeric_cols:
        summary_stats.append({
            'Parameter': col,
            'Mean': df[col].mean(),
            'Min': df[col].min(),
            'Max': df[col].max(),
            'Std': df[col].std(),
            'Count': df[col].count()
        })
    
    return pd.DataFrame(summary_stats)

def analyze_put_pricing_debug(put_pricing_debug):
    if not put_pricing_debug :
        return pd.DataFrame()

    # Convert list of dicts to DataFrame
    df = pd.DataFrame(put_pricing_debug )

    # Select numeric columns to analyze
    numeric_cols = [
        'spot_price', 'put_strike',
        'vol_put', 'tau_expiry', 
        'risk_free', 'div_yield', 'put_price', 
        'quantity', 'pnl'
    ]

    # Filter to only columns that exist in the dataframe
    numeric_cols = [col for col in numeric_cols if col in df.columns]

    summary_stats = []
    for col in numeric_cols:
        summary_stats.append({
            'Parameter': col,
            'Mean': df[col].mean(),
            'Min': df[col].min(),
            'Max': df[col].max(),
            'Std': df[col].std(),
            'Count': df[col].count()
        })

    
    return pd.DataFrame(summary_stats)

def aggregate_pricing_summaries(summaries_list):
    """Aggregate pricing summaries from multiple backtests.
    
    Takes a list of summary DataFrames (one per backtest) and calculates
    mean, min, max across all backtests for each parameter.
    
    Parameters:
        summaries_list: List of DataFrames from analyze_put_pricing_debug or analyze_collar_pricing_debug
        
    Returns:
        DataFrame with aggregated statistics across all backtests
    """
    if not summaries_list:
        return pd.DataFrame()
    
    # Concatenate all summaries
    all_summaries = pd.concat(summaries_list, ignore_index=True)
    
    # Group by Parameter and calculate aggregated statistics
    aggregated_stats = []
    for param in all_summaries['Parameter'].unique():
        param_data = all_summaries[all_summaries['Parameter'] == param]
        aggregated_stats.append({
            'Parameter': param,
            'Mean_of_Means': param_data['Mean'].mean(),
            'Mean_of_Mins': param_data['Min'].mean(),
            'Mean_of_Maxs': param_data['Max'].mean(),
            'Overall_Min': param_data['Min'].min(),
            'Overall_Max': param_data['Max'].max(),
            'Std_of_Means': param_data['Mean'].std(),
            'Num_Backtests': param_data['Parameter'].count()
        })
    
    return pd.DataFrame(aggregated_stats)

def create_summary(backtest_summary):
    rend_annuo = backtest_summary['Rend_Annuo'].mean()
    vol_annuo = backtest_summary['Vol_Annua'].mean()
    min_equity = backtest_summary['Max_Equity'].min()
    max_equity = backtest_summary['Max_Equity'].mean()
    mean_equity = backtest_summary['Mean_Equity'].mean()

    final_capitals = backtest_summary['Final_Capital'].values
    total_rets = backtest_summary['Total_Return'].values
    mean_final = np.mean(final_capitals)
    median_final = np.median(final_capitals)

    min_final = np.min(final_capitals)
    max_final = np.max(final_capitals)

    # Calculate success metrics
    profitable_runs = sum(ret > 0 for ret in total_rets)
    total_runs = len(final_capitals)
    success_rate = (profitable_runs / total_runs) * 100

    # Monthly returns
    mean_return = backtest_summary['Period_Ret%'].mean()
    stdev_return = backtest_summary['Period_Std%'].mean()

    collar = backtest_summary['Collar'].mean()

    # Calculate risk metrics
    mean_max_drawdown = backtest_summary['Max_Drawdown%'].mean()
    worst_drawdown = backtest_summary['Max_Drawdown%'].min()

    summary_df = pd.DataFrame({
        'Rend_Annuo': [rend_annuo],
        'Vol_Annuo': [vol_annuo],
        'Rend/Vol':[rend_annuo / vol_annuo],  
        'Collar': [collar],
        'Min_Equity': [min_equity],
        'Max_Equity': [max_equity],
        'Mean_Equity': [mean_equity],
        'Mean_Final_Capital': [mean_final],
        'Median_Final_Capital': [median_final],
        'Min_Final_Capital': [min_final],
        'Max_Final_Capital': [max_final],
        'Profitable_Runs': [profitable_runs],
        'Total_Runs': [total_runs],
        'Success_Rate(%)': [success_rate],
        'Mean_Period_Return(%)': [mean_return],
        'Mean_Period_Stdev(%)': [stdev_return],
        'Mean_Max_Drawdown(%)': [mean_max_drawdown],
        'Worst_Drawdown(%)': [worst_drawdown]
    })
    return summary_df


class VolHandle:
    """Helper to interpolate implied volatility on a strike-tenor grid.

    Expects a DataFrame with columns:
    - reference_date: valuation date of the surface
    - moneyness: strike/spot (will be rescaled to absolute strike by caller)
    - tenor: string like '30D' that will be converted to integer days
    - implied_vol: volatility level

    The surface is pivoted to a RegularGridInterpolator over (strike, tenor_years),
    with flat extrapolation on strikes and tenors beyond the grid.
    """

    def __init__(self, vol_df: pd.DataFrame) -> None:
        vol_df = vol_df.copy()
        vol_df["tenor"] = vol_df["tenor"].str.replace("D", "").astype(int)
        self.vol_df = vol_df
        self.interp_surface = self._build_surface()

    def _build_surface(self) -> RegularGridInterpolator:
        self.vol_pivot = self.vol_df.pivot(index="moneyness", columns="tenor", values="implied_vol")
        tenors = (self.vol_pivot.columns / 360).to_list()
        strikes = self.vol_pivot.index.to_list()

        self.tenors = tenors
        self.strikes = strikes
        self.max_tenor = max(tenors)
        self.min_strike = min(strikes)
        self.max_strike = max(strikes)

        return RegularGridInterpolator(
            (strikes, tenors),
            self.vol_pivot.values.tolist(),
            bounds_error=False,
            fill_value=None
        )

    def get_vol(self, strike: float, tenor: float) -> float:
        """Return interpolated (or flat-extrapolated) implied volatility for given strike and tenor (in years)."""
        # Flat extrapolation on strikes
        if strike < self.min_strike:
            K = self.min_strike
        elif strike > self.max_strike:
            K = self.max_strike
        else:
            K = strike

        # Flat extrapolation on tenors
        if tenor <= 0:
            return 0
        elif tenor > self.max_tenor:
            T = self.max_tenor
        else:
            T = tenor

        return float(self.interp_surface((K, T)))

def calculate_max_drawdown(df: pd.DataFrame, start_date: date, end_date: date) -> float:
    """Compute maximum drawdown between two dates for a DataFrame with columns `date` and `price`.

    Returns the minimum (most negative) drawdown as a float (e.g., -0.25 for -25%).
    """
    # Filter data up to input_date
    df = df[df['date'] >= start_date].copy()
    df = df[df['date'] <= end_date].copy()
    # Compute running maximum price
    df['running_max'] = df['close'].cummax()
    # Compute drawdown: (current price - max price) / max price
    df['drawdown'] = (df['close'] - df['running_max']) / df['running_max']
    # Find the maximum drawdown (most negative value)
    max_drawdown = df['drawdown'].min()
    return max_drawdown

def first_date_per_month(date_array):
    # Se l'input è un array numpy, lo converto in lista per sicurezza
    dates = list(date_array)
    dates.sort()  # Ordina cronologicamente
    
    first_dates = {}
    for d in dates:
        ym = (d.year, d.month)
        if ym not in first_dates:  # tiene solo la prima data di ogni mese
            first_dates[ym] = d
    
    return np.array(list(first_dates.values()))

def middle_date_per_month(date_array):
    # Converti in lista e ordina cronologicamente
    dates = sorted(list(date_array))
    
    # Raggruppa le date per anno-mese
    month_groups = defaultdict(list)
    for d in dates:
        month_groups[(d.year, d.month)].append(d)
    
    middle_dates = []
    for ym, d_list in month_groups.items():
        n = len(d_list)
        middle_idx = n // 2  # indice centrale (arrotonda per difetto)
        middle_dates.append(d_list[middle_idx])
    
    return np.array(middle_dates)

def calculate_ptf_drawdown(portfolio_values: pd.Series) -> float:
    """Calculate the maximum drawdown for a portfolio value series.

    Returns the drawdown as a negative percentage (e.g., -25 for -25%).
    """
    running_max = portfolio_values.cummax()  # Rolling max up to each point
    drawdowns = (portfolio_values - running_max) / running_max  # Calculate drawdowns
    return drawdowns.min() * 100  # Get the worst drawdown (min value)

def analyze_results(res):
    backtest_results = []
    initial_value = res['Portfolio_Value'].values[0]
    final_value = res['Portfolio_Value'].iloc[-1]
    res_returns = res['Portfolio_Value'].pct_change()[1:]
    total_invested = res['Total_Invested'].values[-1]
    max_equity = (res['Equity part'] / res['Portfolio_Value']).max()
    mean_equity = (res['Equity part'] / res['Portfolio_Value']).mean()
    max_drawdown = calculate_ptf_drawdown(res['Portfolio_Value'])
    try:
        collar_value = res.loc[res['Unwind'] == True, 'Collar PnL'].values[0]
    except IndexError:
        collar_value = res['Collar PnL'].iloc[-1]

    annualized_return = (final_value /100) ** (1/5) -1

    backtest_results.append({
                'Start_Date': res['Start_Date'].values[0],
                'Initial_Price': res['Initial_Price'].iloc[0],
                'Total_Invested': total_invested,
                'Max_Equity': max_equity,
                'Mean_Equity': mean_equity,
                'Final_Capital': final_value,
                'Collar': collar_value,
                'Total_Return': (final_value / initial_value - 1) * 100,
                'Rend_Annuo': annualized_return * 100,
                'Vol_Annua': res_returns.std() * np.sqrt(252) * 100 ,
                'Sharpe': annualized_return / (res_returns.std() * np.sqrt(252)), 
                'Max_Drawdown%': max_drawdown,
                'Period_Ret%': res_returns.mean()*100,
                'Period_Std%': res_returns.std()*100,
                'Sharpe_Period': res_returns.mean() / res_returns.std(), 


        })
        
    return pd.DataFrame(backtest_results)

def option_price(
  valuation_date: date,
  maturity_date: date,
  spot_price: float,
  strike_price: float,
  volatility: float,
  risk_free_rate: float,
  dividend_rate: float,
  option_type: ql.Option,
) -> float:
  """Price a European option using QuantLib's Black-Scholes-Merton analytic engine.

  Parameters are expressed in spot/strike units, annualized rates and vol, and Python `date`s.
  Returns the NPV as a float in the same currency as the spot.
  """
  valuation_date = ql.Date.from_date(valuation_date)
  maturity_date = ql.Date.from_date(maturity_date)
  ql.Settings.instance().evaluationDate = valuation_date
  day_count = ql.Actual365Fixed()
  calendar = ql.TARGET()
  payoff = ql.PlainVanillaPayoff(option_type, strike_price)
  exercise = ql.EuropeanExercise(maturity_date)
  european_option = ql.VanillaOption(payoff, exercise)


  spot_handle = ql.QuoteHandle(
      ql.SimpleQuote(spot_price)
  )
  flat_ts = ql.YieldTermStructureHandle(
      ql.FlatForward(valuation_date, risk_free_rate, day_count)
  )
  dividend_yield = ql.YieldTermStructureHandle(
      ql.FlatForward(valuation_date, dividend_rate, day_count)
  )
  flat_vol_ts = ql.BlackVolTermStructureHandle(
      ql.BlackConstantVol(valuation_date, calendar, volatility, day_count)
  )
  bsm_process = ql.BlackScholesMertonProcess(spot_handle,
                                            dividend_yield,
                                            flat_ts,
                                            flat_vol_ts)
  european_option.setPricingEngine(ql.AnalyticEuropeanEngine(bsm_process))
  return european_option.NPV()



class PutPicPac:

    def __init__(self,
                 data: pd.DataFrame,
                 vol_df: pd.DataFrame,
                 initial_capital: float,
                 investment_characteristics: pd.DataFrame,
                 initial_equity,
                 max_investment: float,
                 investment_horizon: int,
                 put_strikes: float,
                 put_strategy: bool,                
                 ) -> None:
        """Phased equity allocation with dynamic short-put/long-call collar overlay.

        Collars are added alongside new equity allocations and unwound on drawdown triggers.
        """
        self.data = data.copy()
        self.vol_df = vol_df.copy()
        self.initial_capital = initial_capital
        self.investment_characteristics = investment_characteristics
        self.initial_equity = initial_equity
        self.max_investment = max_investment
        self.investment_horizon = investment_horizon
        self.put_strikes = put_strikes
        self.put_strategy = put_strategy

    def get_investment_dates(self, all_dates, investment_characteristics):
        """
        Restituisce una lista di date corrispondenti a t0 + n mesi, 
        dove t0 è la prima data in all_dates.
        Se la data teorica non esiste in all_dates, ritorna la prima successiva disponibile.
        """
        all_dates = sorted(pd.to_datetime(all_dates))
        t0 = all_dates[0]
        result_dates = []
        
        for n in investment_characteristics['Period']:
            target_date = t0 + relativedelta(months=int(n))
            
            # Trova la prima data disponibile in all_dates >= target_date
            next_dates = [d for d in all_dates if d >= target_date]
            if next_dates:
                result_dates.append(next_dates[0].date())
            else:
                result_dates.append(all_dates[-1].date())  # se oltre l’ultima disponibile
        
        return result_dates
    
    def get_investment_schedule(self, all_dates, investment_characteristics):
        """
        Restituisce un dizionario {data: nominal} per ciascun periodo definito.
        """
        investment_dates = self.get_investment_dates(all_dates, investment_characteristics)
        return dict(zip(investment_dates, investment_characteristics['Nominal']))
    
    def run_single_backtest(self, start_date: date, debug: bool = False):
        period_data = self.data[
                    (self.data['date'] >= start_date) &
                    (self.data['date'] <= start_date + timedelta(365* self.investment_horizon) )
                    ].copy()

        all_dates = period_data['date'].values
        end_date = all_dates.max()
        investment_schedule = self.get_investment_schedule(all_dates, self.investment_characteristics)

        daily_data = period_data.reset_index(drop=True)
        initial_price = daily_data.iloc[0]['close']

        results = []
        portfolio_value = self.initial_capital 
        safe_part = self.initial_capital
        total_invested = 0  # Track total invested amount
        risky_part = 0  # Track current investment value

        first_of_month = first_date_per_month(all_dates)

        self.rf_t_grid = np.arange(0.1, 5.1, 0.25)
        self.rf_curves_debug = np.zeros((len(daily_data), len(self.rf_t_grid)))

        positions = {}

        # Initialize debug variables for collar pricing parameters
        if debug:
            self.put_pricing_debug = []  # List to store all collar pricing parameters

        for i in range(len(daily_data)):
            df_tmp = daily_data.iloc[:i+1,:]
            prev_date_period = df_tmp.iloc[i-1,0]
            current_date_period = df_tmp.iloc[i,0]
            tau = (current_date_period - prev_date_period).days/365

            prev_price = df_tmp.iloc[i - 1]['close']
            current_price = df_tmp.iloc[-1]['close']
            performance = current_price / prev_price

            vol_df_tmp = self.vol_df[self.vol_df['date'] == current_date_period].copy()
            vol_df_tmp['moneyness'] *= current_price
            if vol_df_tmp.empty:
                raise ValueError(f"No volatility data found for date: {current_date_period}")
            surface = VolHandle(vol_df_tmp)

            risk_free = df_tmp.iloc[- 1]['rate1']
            risk_free2 = df_tmp.iloc[- 1]['rate2']
            risk_free4 = df_tmp.iloc[- 1]['rate4']
            risk_free5 = df_tmp.iloc[- 1]['rate5']

            rf_times = np.array([1, 2, 4, 5])
            rf_rates = np.array([risk_free, risk_free2, risk_free4, risk_free5])
            # np.interp by default uses left and right values for extrapolation, so for t<1 it returns risk_free
            risk_free_interp = lambda t: float(np.interp(t, rf_times, rf_rates, left=risk_free, right=risk_free5))

            div_yield = df_tmp.iloc[- 1]['div_yield']

            if i == 0: # primo giorno
                new_investment = self.initial_equity
                total_invested += new_investment
                risky_part += new_investment
                safe_part -= new_investment

                # posizioni put:
                put_cash_premium = 0
                if self.put_strategy:
                    opt_anag = pd.DataFrame({'maturity': investment_schedule.keys(), 'strike': self.put_strikes})
                    opt_anag['strike'] = opt_anag['strike'] * current_price
                    opt_anag['size'] = (self.max_investment - self.initial_equity) / len(investment_schedule.keys()) / opt_anag['strike'] 

                    # Registra posizioni e incassa premi
                    total_premium = 0
                    for ix, row in opt_anag.iterrows():
                        tau_expiry = (row['maturity'] - current_date_period).days / 365
                        rf_T = risk_free_interp(tau_expiry) 
                        vol = surface.get_vol(row['strike'], tau_expiry)
                        put_price = option_price(current_date_period, row['maturity'], current_price,
                                                    row['strike'], vol, rf_T, div_yield, ql.Option.Put)
                        premium = put_price * row['size']
                        total_premium += premium

                        positions[f'put_{ix}'] = {
                            'initial_price': put_price,
                            'quantity': row['size'],
                            'closed_quantity': 0,
                            'premium': premium,
                            'trade_date': current_date_period,
                            'maturity': row['maturity'],
                            'strike': row['strike'],
                            'riskfree': rf_T,
                            'divyield': div_yield,
                            'put_strike': row['strike'],
                            'put_vol': vol,
                            'last_price': None,
                            'last_pnl': None
                        }

                        if debug: 
                            self.put_pricing_debug.append({
                                'iteration': i,
                                'date': current_date_period,
                                'event_type': 'initial_put',
                                'key': f'put_{ix}',
                                'spot_price': current_price,
                                'put_strike': row['strike'],
                                'vol_put': vol,
                                'tau_expiry': tau_expiry,
                                'risk_free': rf_T,
                                'div_yield': div_yield,
                                'put_price': premium,
                                'quantity': row['size'],
                                'maturity': row['maturity'],
                                'pnl': 0
                            })

                    put_cash_premium = total_premium # incasso premio cash e investo in bond
                results.append({
                        'Start_Date': prev_date_period,
                        'End_Date': current_date_period,
                        'New_Investment': new_investment,
                        'Current_Investment': risky_part,
                        'Total_Invested': total_invested,
                        'Initial_Price': initial_price,
                        'Index Price': current_price,
                        'Index Return': performance - 1,
                        'Portfolio_Value': portfolio_value,
                        'Bond part': safe_part,
                        'Equity part': risky_part,
                        'Put Value': put_cash_premium,
                        'Put PnL': 0
                    })
                continue

            # Calcola PnL sulle put
            put_pnl = 0
            put_value = 0
            # is_expiry = False
            put_price = 0
            if self.put_strategy:
                for key, pos in positions.items():

                    if current_date_period == pos['maturity']:
                        payoff = max(pos['put_strike'] - current_price, 0)
                        put_price = payoff * pos['quantity']
                        pnl = -(payoff - pos['initial_price']) * pos['quantity'] # negativo se ITM. se OTM positivo (premio incassato per expiry)
                        tau_expiry = 0
                        vol = None
                        rf_T = None
                        pos['last_price'] = put_price
                        pos['last_pnl'] = pnl
                        pos['closed_quantity'] = pos['quantity']
                        pos['quantity'] = 0
                    elif current_date_period > pos['maturity']:
                        tau_expiry = None
                        vol = None
                        rf_T = None
                        put_price = pos['last_price']
                        pnl = pos['last_pnl']
                    else:
                        tau_expiry = (pos['maturity'] - current_date_period).days / 365
                        rf_T = risk_free_interp(tau_expiry) 
                        vol = surface.get_vol(pos['strike'], tau_expiry)
                        put_mtm = option_price(current_date_period, pos['maturity'], current_price,
                                                    pos['strike'], vol, rf_T, div_yield, ql.Option.Put)
                        put_price = put_mtm * pos['quantity']
                        pnl = -(put_mtm - pos['initial_price']) * pos['quantity'] 

                    
                    put_value += put_price
                    put_pnl += pnl

                    if debug: 
                        self.put_pricing_debug.append({
                            'iteration': i,
                            'date': current_date_period,
                            'event_type': f'revaluation',
                            'key': key,
                            'spot_price': current_price,
                            'put_strike': pos['strike'],
                            'vol_put': vol,
                            'tau_expiry': tau_expiry,
                            'risk_free': rf_T,
                            'div_yield': div_yield,
                            'put_price': put_price,
                            'quantity': pos['quantity'],
                            'maturity': pos['maturity'],
                            'pnl': pnl
                        })
                    
            # Aggiorna parte bond (crescita con interesse risk-free a t1)
            safe_part *= (1 + (risk_free * tau))
            # Aggiorna parte equity
            risky_part *= performance
            risky_log = risky_part

            # Valore portafoglio aggiornato
            portfolio_value = safe_part + risky_part + put_pnl

            
            # Aggiorna la posizione Risky alle date di investimento
            new_investment = 0
            if total_invested < self.max_investment:
                if current_date_period in investment_schedule.keys(): # se siamo in una data di investimento
                    new_investment = investment_schedule[current_date_period] * (self.max_investment - self.initial_equity)
                    total_invested += new_investment
                    risky_part += new_investment
                    safe_part -= new_investment

            results.append({
                'Start_Date': prev_date_period,
                'End_Date': current_date_period,
                'New_Investment': new_investment,
                'Current_Investment': risky_part,
                'Total_Invested': total_invested,
                'Initial_Price': initial_price,
                'Index Price': current_price,
                'Index Return': performance - 1,
                'Portfolio_Value': portfolio_value,
                'Bond part': safe_part,
                'Equity part': risky_log,
                'Put Value': put_value,
                'Put PnL': put_pnl
            })
        results = pd.DataFrame(results)
        positions = pd.DataFrame(positions)
        return results, positions
    
    def run_rolling_backtests(self, frequency= "1M", batch = 'benchmark', debug: bool = False):
        """Run rolling backtests starting each month.

        Returns a concatenated results DataFrame and a dict of positions per start date.
        """
        all_results = []
        # Initialize collar_pricing_debug to collect data from all backtests
        if debug:
            self.collar_pricing_debug = []
        
        start_dates = self.data['date'].values
        if frequency == "1M":
            start_dates = first_date_per_month(start_dates)
        else:
            start_dates = middle_date_per_month(start_dates)
        
        # i = 0
        for start_date in start_dates:
            print('## CALCULATING: ', start_date)
            # i += 1
            # if i == 2:
            #     break
            

            if start_date + timedelta(365* self.investment_horizon) > self.data['date'].max():
                break

            result, _ = self.run_single_backtest(start_date, debug)
            if result is not None:
                summary = analyze_results_put(result)
                all_results.append(summary)
        
        res_out = pd.concat(all_results, ignore_index=True)
        
        # Export results to Excel with multiple sheets
        output_file = f"results/{batch}_results.xlsx"
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            res_out.to_excel(writer, sheet_name='Results', index=False)
            
            # Calculate and export collar pricing debug summary if available
            if debug and hasattr(self, 'put_pricing_debug') and self.put_pricing_debug:
                put_summary = analyze_put_pricing_debug(self.put_pricing_debug)
                put_summary.to_excel(writer, sheet_name='Put_Pricing_Stats', index=False)

        return res_out


class DynamicFund:

    def __init__(self,
                 data: pd.DataFrame,
                 vol_df: pd.DataFrame,
                 initial_capital: float,
                 investment_characteristics: pd.DataFrame,
                 initial_equity,
                 max_investment: float,
                 investment_horizon: int,
                 put_strategy: bool,   
                 put_strikes: float,
                 put_strike: float,
                 call_strike: float,
                 collar_strategy: bool,
                 drawdown_treshold: float,
                 eom: bool,
                 ) -> None:
        """Phased equity allocation with dynamic short-put/long-call collar overlay.

        Collars are added alongside new equity allocations and unwound on drawdown triggers.
        """
        self.data = data.copy()
        self.vol_df = vol_df.copy()
        self.initial_capital = initial_capital
        self.investment_characteristics = investment_characteristics
        self.initial_equity = initial_equity
        self.max_investment = max_investment
        self.investment_horizon = investment_horizon
        self.put_strikes = put_strikes
        self.put_strategy = put_strategy
        self.collar_low = put_strike
        self.collar_up = call_strike
        self.collar_strategy = collar_strategy#
        self.drawdown_treshold = drawdown_treshold
        self.eom = eom

    def get_investment_dates(self, all_dates, investment_characteristics):
        """
        Restituisce una lista di date corrispondenti a t0 + n mesi, 
        dove t0 è la prima data in all_dates.
        Se la data teorica non esiste in all_dates, ritorna la prima successiva disponibile.
        """
        all_dates = sorted(pd.to_datetime(all_dates))
        t0 = all_dates[0]
        result_dates = []
        
        for n in investment_characteristics['Period']:
            target_date = t0 + relativedelta(months=int(n))
            
            # Trova la prima data disponibile in all_dates >= target_date
            next_dates = [d for d in all_dates if d >= target_date]
            if next_dates:
                result_dates.append(next_dates[0].date())
            else:
                result_dates.append(all_dates[-1].date())  # se oltre l’ultima disponibile
        
        return result_dates
    
    def get_investment_schedule(self, all_dates, investment_characteristics):
        """
        Restituisce un dizionario {data: nominal} per ciascun periodo definito.
        """
        investment_dates = self.get_investment_dates(all_dates, investment_characteristics)
        return dict(zip(investment_dates, investment_characteristics['Nominal']))
    
    def run_single_backtest(self, start_date: date, debug: bool = False):
        period_data = self.data[
                (self.data['date'] >= start_date) &
                (self.data['date'] <= start_date + timedelta(365* self.investment_horizon) )
                ].copy()

        all_dates = period_data['date'].values
        end_date = all_dates.max()
        investment_schedule = self.get_investment_schedule(all_dates, self.investment_characteristics)

        daily_data = period_data.reset_index(drop=True)
        initial_price = daily_data.iloc[0]['close']

        results = []
        portfolio_value = self.initial_capital 
        safe_part = self.initial_capital
        total_invested = 0  # Track total invested amount
        risky_part = 0  # Track current investment value

        first_of_month = first_date_per_month(all_dates)

        self.rf_t_grid = np.arange(0.1, 5.1, 0.25)
        self.rf_curves_debug = np.zeros((len(daily_data), len(self.rf_t_grid)))

        # Strategy pmts
        drawdown_condition = False 
        add_unwind = False
        collar_pnl = 0
        unwinded_pnl = 0
        collar_positions = {}
        put_positions = {}

        # Initialize debug lists
        if debug:
            if not hasattr(self, 'put_pricing_debug'):
                self.put_pricing_debug = []
            if not hasattr(self, 'collar_pricing_debug'):
                self.collar_pricing_debug = []

        for i in range(len(daily_data)):
            df_tmp = daily_data.iloc[:i+1,:]
            prev_date_period = df_tmp.iloc[i-1,0]
            current_date_period = df_tmp.iloc[i,0]
            tau = (current_date_period - prev_date_period).days/365

            # Price index
            prev_price = df_tmp.iloc[i - 1]['close']
            current_price = df_tmp.iloc[-1]['close']
            performance = current_price / prev_price

            # total return index
            prev_price_tr = df_tmp.iloc[i - 1]['close_tr']
            current_price_tr = df_tmp.iloc[-1]['close_tr']
            performance_tr = current_price_tr / prev_price_tr

            vol_df_tmp = self.vol_df[self.vol_df['date'] == current_date_period].copy()
            vol_df_tmp['moneyness'] *= current_price
            if vol_df_tmp.empty:
                raise ValueError(f"No volatility data found for date: {current_date_period}")
            surface = VolHandle(vol_df_tmp)

            risk_free = df_tmp.iloc[- 1]['rate1']
            risk_free2 = df_tmp.iloc[- 1]['rate2']
            risk_free4 = df_tmp.iloc[- 1]['rate4']
            risk_free5 = df_tmp.iloc[- 1]['rate5']

            rf_times = np.array([1, 2, 4, 5])
            rf_rates = np.array([risk_free, risk_free2, risk_free4, risk_free5])
            # np.interp by default uses left and right values for extrapolation, so for t<1 it returns risk_free
            risk_free_interp = lambda t: float(np.interp(t, rf_times, rf_rates, left=risk_free, right=risk_free5))

            div_yield = df_tmp.iloc[- 1]['div_yield']

            if i == 0: # primo giorno
                new_investment = self.initial_equity
                total_invested += new_investment
                risky_part += new_investment
                safe_part -= new_investment

                # posizioni put:
                put_cash_premium = 0

                if self.put_strategy:
                    opt_anag = pd.DataFrame({'maturity': investment_schedule.keys(), 'strike': self.put_strikes})
                    opt_anag['strike'] = opt_anag['strike'] * current_price
                    opt_anag['size'] = (self.max_investment - self.initial_equity) / len(investment_schedule.keys()) / opt_anag['strike'] 

                    # Registra posizioni e incassa premi
                    total_premium = 0
                    for ix, row in opt_anag.iterrows():
                        tau_expiry = (row['maturity'] - current_date_period).days / 365
                        rf_T = risk_free_interp(tau_expiry) 
                        vol = surface.get_vol(row['strike'], tau_expiry)
                        put_price = option_price(current_date_period, row['maturity'], current_price,
                                                    row['strike'], vol, rf_T, div_yield, ql.Option.Put)
                        premium = put_price * row['size']
                        total_premium += premium

                        put_positions[f'put_{ix}'] = {
                            'initial_price': put_price,
                            'quantity': row['size'],
                            'closed_quantity': 0,
                            'premium': premium,
                            'trade_date': current_date_period,
                            'maturity': row['maturity'],
                            'strike': row['strike'],
                            'riskfree': rf_T,
                            'divyield': div_yield,
                            'put_strike': row['strike'],
                            'put_vol': vol,
                            'last_price': None,
                            'last_pnl': None
                        }

                        if debug: 
                            self.put_pricing_debug.append({
                                'iteration': i,
                                'date': current_date_period,
                                'event_type': 'initial_put',
                                'key': f'put_{ix}',
                                'spot_price': current_price,
                                'put_strike': row['strike'],
                                'vol_put': vol,
                                'tau_expiry': tau_expiry,
                                'risk_free': rf_T,
                                'div_yield': div_yield,
                                'put_price': premium,
                                'quantity': row['size'],
                                'maturity': row['maturity'],
                                'pnl': 0
                            })

                    put_cash_premium = total_premium # incasso premio cash e investo in bond

                # posizioni collar:
                if self.collar_strategy:
                    collar_quantity = new_investment/current_price
                    tau_expiry = (end_date-current_date_period).days/365 # time to maturity, da data di valutazione a end_date
                    rf_T = risk_free_interp(tau_expiry) 
                    vol_put = surface.get_vol(self.collar_low * current_price, tau_expiry) # vola implicita put
                    vol_call = surface.get_vol(self.collar_up * current_price, tau_expiry) # vola implicita call
                    put = option_price(current_date_period, end_date, current_price, self.collar_low*current_price, 
                                                vol_put, rf_T, div_yield, ql.Option.Put)
                    call = option_price(current_date_period, end_date, current_price, self.collar_up*current_price,
                                        vol_call, rf_T, div_yield, ql.Option.Call)
                    
                    collar_value = put - call 
                    collar_positions[f'collar_{i}'] = {
                                                    'initial_price': collar_value,
                                                    'quantity': collar_quantity,
                                                    'trade_date': current_date_period,
                                                    'maturity': end_date,
                                                    'tau': tau_expiry,
                                                    'put_price': put,
                                                    'call_price': call,
                                                    'risk_free': rf_T,
                                                    'div_yield': div_yield,
                                                    'put_strike': self.collar_low*current_price,
                                                    'put_vol': vol_put,
                                                    'call_strike': self.collar_up*current_price,
                                                    'call_vol': vol_call,
                                                }
                    
                    # Save collar pricing parameters for debug
                    if debug:
                        self.collar_pricing_debug.append({
                            'iteration': i,
                            'date': current_date_period,
                            'event_type': 'initial_collar',
                            'spot_price': current_price,
                            'put_strike': self.collar_low*current_price,
                            'call_strike': self.collar_up*current_price,
                            'vol_put': vol_put,
                            'vol_call': vol_call,
                            'tau_expiry': tau_expiry,
                            'risk_free': rf_T,
                            'div_yield': div_yield,
                            'put_price': put,
                            'call_price': call,
                            'collar_value': collar_value,
                            'collar_quantity': collar_quantity,
                            'maturity': end_date
                        })

                # Salva risultati
                results.append({
                    'Start_Date': prev_date_period,
                    'End_Date': current_date_period,
                    'New_Investment': new_investment,
                    'Current_Investment': risky_part,
                    'Total_Invested': total_invested,
                    'Initial_Price': initial_price,
                    'Index Price': current_price,
                    'Index Return': performance - 1,
                    'Portfolio_Value': portfolio_value,
                    'Bond part': safe_part,
                    'Equity part': risky_part,
                    'Collar PnL': collar_pnl,
                    'Unwind': add_unwind,
                    'Put Value': put_cash_premium,
                    'Put PnL': 0
                })
                continue

            ######## RIVALUTAZIONI A OGNI T
            collar_pnl = 0
            collar_net_position = 0
            if self.collar_strategy and not drawdown_condition:
                for key, value in collar_positions.items():
                    if current_date_period == end_date:
                        put = max(value['put_strike'] - current_price, 0)
                        call = max(current_price - value['call_strike'], 0)
                        final_pnl = ((put - call) - value['initial_price']) * value['quantity'] 
                        collar_pnl += final_pnl
                        continue
                    
                    tau_expiry = (end_date-current_date_period).days/365
                    rf_T = risk_free_interp(tau_expiry) 
                    vol_put = surface.get_vol(value['put_strike'], tau_expiry)
                    vol_call = surface.get_vol(value['call_strike'], tau_expiry)
                    put = option_price(current_date_period, end_date, current_price, value['put_strike'], 
                                        vol_put, rf_T, div_yield, ql.Option.Put)
                    call = option_price(current_date_period, end_date, current_price, value['call_strike'],
                                        vol_call,  rf_T, div_yield, ql.Option.Call)
                    value[f'pnl_M{i}'] = ((put - call) - value['initial_price']) * value['quantity'] 

                    collar_pnl += value[f'pnl_M{i}']
                    collar_net_position += value['quantity']
                    
                    # Save collar pricing parameters for debug
                    if debug:
                        self.collar_pricing_debug.append({
                            'iteration': i,
                            'date': current_date_period,
                            'event_type': 'revaluation',
                            'collar_key': key,
                            'spot_price': current_price,
                            'put_strike': value['put_strike'],
                            'call_strike': value['call_strike'],
                            'vol_put': vol_put,
                            'vol_call': vol_call,
                            'tau_expiry': tau_expiry,
                            'risk_free': rf_T,
                            'div_yield': div_yield,
                            'put_price': put,
                            'call_price': call,
                            'collar_value': put - call,
                            'collar_quantity': value['quantity'],
                            'pnl': value[f'pnl_M{i}'],
                            'maturity': end_date
                        })

            # se c'è stato unwind all'iterazione precedente si aggiorna la risky part:
            if add_unwind:
                # print("prev risky: ", risky_part)
                risky_part += unwinded_pnl
                # print("new risky: ", risky_part)
                add_unwind = False

            put_pnl = 0
            put_value = 0
            put_price = 0
            if self.put_strategy:
                for key, pos in put_positions.items():

                    if current_date_period == pos['maturity']:
                        payoff = max(pos['put_strike'] - current_price, 0)
                        put_price = payoff * pos['quantity']
                        pnl = -(payoff - pos['initial_price']) * pos['quantity'] # negativo se ITM. se OTM positivo (premio incassato per expiry)
                        tau_expiry = 0
                        vol = None
                        rf_T = None
                        pos['last_price'] = put_price
                        pos['last_pnl'] = pnl
                        pos['closed_quantity'] = pos['quantity']
                        pos['quantity'] = 0
                    elif current_date_period > pos['maturity']:
                        tau_expiry = None
                        vol = None
                        rf_T = None
                        put_price = pos['last_price']
                        pnl = pos['last_pnl']
                    else:
                        tau_expiry = (pos['maturity'] - current_date_period).days / 365
                        rf_T = risk_free_interp(tau_expiry) 
                        vol = surface.get_vol(pos['strike'], tau_expiry)
                        put_mtm = option_price(current_date_period, pos['maturity'], current_price,
                                                    pos['strike'], vol, rf_T, div_yield, ql.Option.Put)
                        put_price = put_mtm * pos['quantity']
                        pnl = -(put_mtm - pos['initial_price']) * pos['quantity'] 

                    
                    put_value += put_price
                    put_pnl += pnl

                    if debug: 
                        self.put_pricing_debug.append({
                            'iteration': i,
                            'date': current_date_period,
                            'event_type': f'revaluation',
                            'key': key,
                            'spot_price': current_price,
                            'put_strike': pos['strike'],
                            'vol_put': vol,
                            'tau_expiry': tau_expiry,
                            'risk_free': rf_T,
                            'div_yield': div_yield,
                            'put_price': put_price,
                            'quantity': pos['quantity'],
                            'maturity': pos['maturity'],
                            'pnl': pnl
                        })
            # Aggiorna parte bond (crescita con interesse risk-free a t1)
            safe_part *= (1 + (risk_free * tau))

            # Aggiorna parte equity
            risky_part *= performance_tr
            risky_log = risky_part

            # Valore portafoglio aggiornato
            portfolio_value = safe_part + risky_part + collar_pnl + put_pnl

            # Trigger di drawdown dell'indice 
            # quando l'indice supera la soglia di drawdown vengono unwindati tutti i collar 
            if self.collar_strategy and not drawdown_condition:
                if self.eom and current_date_period in first_of_month:
                    running_max = df_tmp['close'].max()
                    current_price_drawdown = (current_price - running_max) / running_max

                    # Triggera unwind solo se il prezzo è ancora sotto la soglia alla data first_of_month
                    if abs(current_price_drawdown) >= self.drawdown_treshold:
                        print("found dd: ", current_date_period)
                        print("*")
                        for key, value in collar_positions.items():
                            value['quantity'] = 0
                        drawdown_condition = True 
                        unwinded_pnl =  collar_pnl # si salva il pnl chiuso in questa variabile che al loop successivo viene aggiunta al portafoglio
                        add_unwind = True # salva il flag così verrà inserito il capitale di unwind alla prossima iterazione 
                elif not self.eom:
                    running_max = df_tmp['close'].max()
                    current_price_drawdown = (current_price - running_max) / running_max

                    # Triggera unwind solo se il prezzo è ancora sotto la soglia alla data first_of_month
                    if abs(current_price_drawdown) >= self.drawdown_treshold:
                        print("found dd daily: ", current_date_period)
                        print("*")
                        for key, value in collar_positions.items():
                            value['quantity'] = 0
                        drawdown_condition = True 
                        unwinded_pnl =  collar_pnl # si salva il pnl chiuso in questa variabile che al loop successivo viene aggiunta al portafoglio
                        add_unwind = True # salva il flag così verrà inserito il capitale di unwind alla prossima iterazione 
                    

            # Aggiorna la posizione Risky alle date di investimento
            new_investment = 0
            if total_invested < self.max_investment:
                if current_date_period in investment_schedule.keys(): # se siamo in una data di investimento
                    new_investment = investment_schedule[current_date_period] * (self.max_investment - self.initial_equity)
                    total_invested += new_investment
                    risky_part += new_investment
                    safe_part -= new_investment

                    if not drawdown_condition and self.collar_strategy:
                        # posizioni collar
                        collar_quantity = new_investment/current_price
                        tau_expiry = (end_date-current_date_period).days/365 # time to maturity, da data di valutazione a end_date
                        rf_T = risk_free_interp(tau_expiry) 
                        vol_put = surface.get_vol(self.collar_low * current_price, tau_expiry) # vola implicita put
                        vol_call = surface.get_vol(self.collar_up * current_price, tau_expiry) # vola implicita call
                        put = option_price(current_date_period, end_date, current_price, self.collar_low*current_price, 
                                                    vol_put, rf_T, div_yield, ql.Option.Put)
                        call = option_price(current_date_period, end_date, current_price, self.collar_up*current_price,
                                            vol_call, rf_T, div_yield, ql.Option.Call)
                        collar_value = put - call 
                        collar_positions[f'collar_{i}'] = {
                                                        'initial_price': collar_value,
                                                        'quantity': collar_quantity,
                                                        'trade_date': current_date_period,
                                                        'maturity': end_date,
                                                        'tau': tau_expiry,
                                                        'put_strike': self.collar_low*current_price,
                                                        'put_vol': vol_put,
                                                        'call_strike': self.collar_up*current_price,
                                                        'call_vol': vol_call,
                                                    }
                        
                        # Save collar pricing parameters for debug
                        if debug:
                            self.collar_pricing_debug.append({
                                'iteration': i,
                                'date': current_date_period,
                                'event_type': 'new_investment_collar',
                                'spot_price': current_price,
                                'put_strike': self.collar_low*current_price,
                                'call_strike': self.collar_up*current_price,
                                'vol_put': vol_put,
                                'vol_call': vol_call,
                                'tau_expiry': tau_expiry,
                                'risk_free': rf_T,
                                'div_yield': div_yield,
                                'put_price': put,
                                'call_price': call,
                                'collar_value': collar_value,
                                'collar_quantity': collar_quantity,
                                'maturity': end_date
                            })

            # Salva risultati
            results.append({
                'Start_Date': prev_date_period,
                'End_Date': current_date_period,
                'New_Investment': new_investment,
                'Current_Investment': risky_part,
                'Total_Invested': total_invested,
                'Initial_Price': initial_price,
                'Index Price': current_price,
                'Index Return': performance - 1,
                'Portfolio_Value': portfolio_value,
                'Bond part': safe_part,
                'Equity part': risky_log,
                'Collar PnL': collar_pnl,
                'Unwind': add_unwind,
                'Put Value': put_value,
                'Put PnL': put_pnl
            })
        results = pd.DataFrame(results)
        collar_positions = pd.DataFrame(collar_positions)
        put_positions = pd.DataFrame(put_positions)
        return results, collar_positions, put_positions

    def run_rolling_backtests(self, frequency= "1M", batch = 'benchmark', debug: bool = False):
        """Run rolling backtests starting each month.

        Returns a concatenated results DataFrame and a dict of positions per start date.
        """
        all_results = []
        all_put_summaries = []  # Store put pricing summaries from each backtest
        all_collar_summaries = []  # Store collar pricing summaries from each backtest
        
        start_dates = self.data['date'].values
        if frequency == "1M":
            start_dates = first_date_per_month(start_dates)
        else:
            start_dates = middle_date_per_month(start_dates)
        
        i = 0
        for start_date in start_dates:
            print('## CALCULATING: ', start_date)
            i += 1
            if i == 3:
                break
            if start_date + timedelta(365* self.investment_horizon) > self.data['date'].max():
                break

            result, _, _ = self.run_single_backtest(start_date, debug)
            if result is not None:
                summary = analyze_dynamic_results(result)
                all_results.append(summary)
                
                # Calculate and store pricing summaries for this backtest
                if debug:
                    if hasattr(self, 'put_pricing_debug') and self.put_pricing_debug:
                        put_summary = analyze_put_pricing_debug(self.put_pricing_debug)
                        put_summary['start_date'] = start_date  # Add identifier
                        all_put_summaries.append(put_summary)
                        self.put_pricing_debug = []  # Reset for next backtest
                    
                    if hasattr(self, 'collar_pricing_debug') and self.collar_pricing_debug:
                        collar_summary = analyze_collar_pricing_debug(self.collar_pricing_debug)
                        collar_summary['start_date'] = start_date  # Add identifier
                        all_collar_summaries.append(collar_summary)
                        self.collar_pricing_debug = []  # Reset for next backtest
        
        res_out = pd.concat(all_results, ignore_index=True)
        
        # Export results to Excel with multiple sheets
        output_file = f"results/{batch}_results.xlsx"
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            res_out.to_excel(writer, sheet_name='Results', index=False)
            
            # Calculate and export aggregated pricing statistics from all backtests
            if debug:
                if all_put_summaries:
                    put_aggregated = aggregate_pricing_summaries(all_put_summaries)
                    put_aggregated.to_excel(writer, sheet_name='Put_Pricing_Stats', index=False)
                
                if all_collar_summaries:
                    collar_aggregated = aggregate_pricing_summaries(all_collar_summaries)
                    collar_aggregated.to_excel(writer, sheet_name='Collar_Pricing_Stats', index=False)
        return res_out

def analyze_dynamic_results(res):
    backtest_results = []
    initial_value = res['Portfolio_Value'].values[0]
    final_value = res['Portfolio_Value'].iloc[-1]
    res_returns = res['Portfolio_Value'].pct_change()[1:]
    total_invested = res['Total_Invested'].values[-1]
    max_equity = (res['Equity part'] / res['Portfolio_Value']).max()
    mean_equity = (res['Equity part'] / res['Portfolio_Value']).mean()
    max_drawdown = calculate_ptf_drawdown(res['Portfolio_Value'])
    try:
        collar_value = res.loc[res['Unwind'] == True, 'Collar PnL'].values[0]
    except IndexError:
        collar_value = res['Collar PnL'].iloc[-1]

    put_upfront_premium = res.iloc[0,:]['Put Value']
    put_pnl = res.iloc[-1,:]['Put PnL']

    annualized_return = (final_value /100) ** (1/5) -1

    backtest_results.append({
                'Start_Date': res['Start_Date'].values[0],
                'Initial_Price': res['Initial_Price'].iloc[0],
                'Total_Invested': total_invested,
                'Max_Equity': max_equity,
                'Mean_Equity': mean_equity,
                'Final_Capital': final_value,
                'Collar': collar_value,
                'Put Upfront': put_upfront_premium,
                'Put PnL': put_pnl,
                'Total_Return': (final_value / initial_value - 1) * 100,
                'Rend_Annuo': annualized_return * 100,
                'Vol_Annua': res_returns.std() * np.sqrt(252) * 100 ,
                'Sharpe': annualized_return / (res_returns.std() * np.sqrt(252)), 
                'Max_Drawdown%': max_drawdown,
                'Period_Ret%': res_returns.mean()*100,
                'Period_Std%': res_returns.std()*100,
                'Sharpe_Period': res_returns.mean() / res_returns.std()
        })
        
    return pd.DataFrame(backtest_results)

def create_summary_all(backtest_summary):
    rend_annuo = backtest_summary['Rend_Annuo'].mean()
    vol_annuo = backtest_summary['Vol_Annua'].mean()
    min_equity = backtest_summary['Max_Equity'].min()
    max_equity = backtest_summary['Max_Equity'].mean()
    mean_equity = backtest_summary['Mean_Equity'].mean()

    final_capitals = backtest_summary['Final_Capital'].values
    total_rets = backtest_summary['Total_Return'].values
    mean_final = np.mean(final_capitals)
    median_final = np.median(final_capitals)

    min_final = np.min(final_capitals)
    max_final = np.max(final_capitals)

    # Calculate success metrics
    profitable_runs = sum(ret > 0 for ret in total_rets)
    total_runs = len(final_capitals)
    success_rate = (profitable_runs / total_runs) * 100

    # Monthly returns
    mean_return = backtest_summary['Period_Ret%'].mean()
    stdev_return = backtest_summary['Period_Std%'].mean()

    collar = backtest_summary['Collar'].mean()
    put = backtest_summary['Put PnL'].mean()

    # Calculate risk metrics
    mean_max_drawdown = backtest_summary['Max_Drawdown%'].mean()
    worst_drawdown = backtest_summary['Max_Drawdown%'].min()

    summary_df = pd.DataFrame({
        'Rend_Annuo': [rend_annuo],
        'Vol_Annuo': [vol_annuo],
        'Rend/Vol':[rend_annuo / vol_annuo],  
        'Collar': [collar],
        'PutPnL': [put],
        'Min_Equity': [min_equity],
        'Max_Equity': [max_equity],
        'Mean_Equity': [mean_equity],
        'Mean_Final_Capital': [mean_final],
        'Median_Final_Capital': [median_final],
        'Min_Final_Capital': [min_final],
        'Max_Final_Capital': [max_final],
        'Profitable_Runs': [profitable_runs],
        'Total_Runs': [total_runs],
        'Success_Rate(%)': [success_rate],
        'Mean_Period_Return(%)': [mean_return],
        'Mean_Period_Stdev(%)': [stdev_return],
        'Mean_Max_Drawdown(%)': [mean_max_drawdown],
        'Worst_Drawdown(%)': [worst_drawdown]
    })
    return summary_df