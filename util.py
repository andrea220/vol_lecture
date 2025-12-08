from collections import defaultdict
import pandas as pd
from datetime import date, timedelta
import QuantLib as ql
from dateutil.relativedelta import relativedelta
import numpy as np
from scipy.interpolate import RegularGridInterpolator

def get_next_available_date(all_dates, start_date, period):
    """
    Ritorna la prima data in all_dates che sia >= start_date + period (in giorni).
    Se non disponibile, ritorna None.
    """
    target_date = start_date + timedelta(days=period)
    # Assicura che all_dates sia ordinato
    sorted_dates = sorted(all_dates)
    for d in sorted_dates:
        if d >= target_date:
            return d
    return None

class StraddleSelling:

    def __init__(self,
                 data: pd.DataFrame,
                 vol_df: pd.DataFrame,
                 initial_capital: float,
                 max_positions: int,
                 ) -> None:
        self.data = data.copy()
        self.vol_df = vol_df.copy()
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.all_dates = self.data['date'].values
        self.start_date = self.data['date'].min()
        self.end_date = self.data['date'].max()

    def run_backtest(self, debug=False):
        portfolio_value = self.initial_capital
        margin = self.initial_capital
        positions = {}
        stock_position = 0  # numero di azioni per hedging
        realized_pnl = 0  # PnL realizzato da posizioni chiuse
        cumulative_stock_pnl = 0  # PnL cumulativo dello stock hedging
        can_open_new_positions = True  # Flag per controllare se possiamo aprire nuove posizioni
        results = []
        
        # P&L decomposition terms (cumulative)
        gamma_term_cum = 0.0
        vega_term_cum = 0.0
        dgamma_term_cum = 0.0
        residual_term_cum = 0.0

        for i in range(len(self.data)):
            # if i == 100:
            #     break
            df_tmp = self.data.iloc[:i+1,:]
            current_date_period = df_tmp.iloc[i,0]

            prev_price = df_tmp.iloc[i - 1]['close']
            current_price = df_tmp.iloc[-1]['close']
            performance = current_price / prev_price

            # Vols
            vol_df_tmp = self.vol_df[self.vol_df['date'] == current_date_period].copy()
            vol_df_tmp['moneyness'] *= current_price
            if vol_df_tmp.empty:
                raise ValueError(f"No volatility data found for date: {current_date_period}")
            surface = VolHandle(vol_df_tmp)

            # rates
            risk_free = df_tmp.iloc[- 1]['rate1']
            risk_free2 = df_tmp.iloc[- 1]['rate2']
            risk_free4 = df_tmp.iloc[- 1]['rate4']
            risk_free5 = df_tmp.iloc[- 1]['rate5']
            rf_times = np.array([1, 2, 4, 5])
            rf_rates = np.array([risk_free, risk_free2, risk_free4, risk_free5])
            risk_free_interp = lambda t: float(np.interp(t, rf_times, rf_rates, left=risk_free, right=risk_free5))
            div_yield = df_tmp.iloc[- 1]['div_yield']

            straddle_pnl = 0  # PnL unrealized (mark-to-market) delle posizioni attive
            straddle_value = 0  # valore mark-to-market totale degli straddle
            straddle_delta = 0
            straddle_gamma = 0
            straddle_vega = 0
            straddle_volga = 0
            active_positions = 0
            
            # revaluate all options
            for key, value in positions.items():
                if value['quantity'] == 0:  # posizione già chiusa
                    continue
                    
                active_positions += 1
                
                if current_date_period == value['maturity']:
                    # SCADENZA: realizziamo il PnL
                    put = max(value['strike'] - current_price, 0)
                    call = max(current_price - value['strike'], 0)
                    current_straddle_value = put + call
                    delta = 0
                    pnl = (current_straddle_value - value['initial_price']) * value['quantity'] * value['side']
                    
                    # Trasferisce il PnL a realized_pnl
                    realized_pnl += pnl

                    value['last_price'] = current_straddle_value
                    value['last_pnl'] = pnl
                    value['closed_quantity'] = value['quantity']
                    value['quantity'] = 0
                    
                    # NON aggiungere a straddle_pnl (è già in realized_pnl)
                    
                else:
                    # Posizione ancora attiva: PnL unrealized
                    tau_expiry = (value['maturity']-current_date_period).days/365
                    rf_T = risk_free_interp(tau_expiry) 
                    vol = surface.get_vol(value['strike'], tau_expiry) 
                    
                    # Get greeks for both put and call
                    put_greeks = option_price(current_date_period, value['maturity'], current_price, value['strike'], 
                                            vol, rf_T, div_yield, ql.Option.Put, return_greeks=True)
                    call_greeks = option_price(current_date_period, value['maturity'], current_price, value['strike'],
                                            vol, rf_T, div_yield, ql.Option.Call, return_greeks=True)
                    
                    put = put_greeks['npv']
                    call = call_greeks['npv']
                    delta_put = put_greeks['delta']
                    delta_call = call_greeks['delta']
                    
                    current_straddle_value = put + call
                    pnl = (current_straddle_value - value['initial_price']) * value['quantity'] * value['side'] 
                    delta = (delta_put + delta_call) * value['quantity'] * value['side']
                    
                    # Aggregate greeks for straddle (put + call)
                    gamma = (put_greeks['gamma'] + call_greeks['gamma']) * value['quantity'] * abs(value['side'])
                    vega = (put_greeks['vega'] + call_greeks['vega']) * value['quantity'] * abs(value['side'])
                    volga = (put_greeks['volga'] + call_greeks['volga']) * value['quantity'] * abs(value['side'])
                    
                    # Store current greeks in position
                    value['current_gamma'] = gamma
                    value['current_vega'] = vega
                    value['current_volga'] = volga
                    value['current_vol'] = vol
                    value['current_tau'] = tau_expiry
                    
                    # Aggiungi a straddle_pnl (unrealized)
                    straddle_pnl += pnl
                    straddle_value += current_straddle_value * value['quantity'] * value['side']
                    straddle_delta += delta
                    straddle_gamma += gamma
                    straddle_vega += vega
                    straddle_volga += volga

            # open new position if needed (mantieni sempre max_positions attive)
            if active_positions < self.max_positions and can_open_new_positions:
                expiry_date = get_next_available_date(self.all_dates, current_date_period, 30)
                
                # Se non ci sono date disponibili, blocca l'apertura di nuove posizioni
                if expiry_date is None:
                    can_open_new_positions = False
                    if debug:
                        print(f"Warning: No expiry date available at {current_date_period}. Stopping new positions.")
                else:
                    tau_expiry = (expiry_date-current_date_period).days/365 
                    rf_T = risk_free_interp(tau_expiry) 
                    strike = current_price
                    vol = surface.get_vol(strike, tau_expiry)
                    
                    # Get greeks for new position
                    put_greeks = option_price(current_date_period, expiry_date, current_price, strike, 
                                            vol, rf_T, div_yield, ql.Option.Put, return_greeks=True)
                    call_greeks = option_price(current_date_period, expiry_date, current_price, strike,
                                            vol, rf_T, div_yield, ql.Option.Call, return_greeks=True)
                    
                    put = put_greeks['npv']
                    call = call_greeks['npv']
                    delta_put = put_greeks['delta']
                    delta_call = call_greeks['delta']

                    new_straddle_value = put + call
                    straddle_quantity = (margin/self.max_positions) / current_price
                    side = -1
                    new_delta = (delta_put + delta_call) * straddle_quantity * side
                    
                    # Aggregate greeks for new straddle
                    new_gamma = (put_greeks['gamma'] + call_greeks['gamma']) * straddle_quantity * abs(side)
                    new_vega = (put_greeks['vega'] + call_greeks['vega']) * straddle_quantity * abs(side)
                    new_volga = (put_greeks['volga'] + call_greeks['volga']) * straddle_quantity * abs(side)
                    
                    # AGGIUNGE al delta totale (non sovrascrive!)
                    straddle_delta += new_delta
                    straddle_pnl += 0  # nuova posizione, pnl = 0
                    straddle_value += new_straddle_value * straddle_quantity * side
                    straddle_gamma += new_gamma
                    straddle_vega += new_vega
                    straddle_volga += new_volga
                    
                    positions[f'straddle_{i}'] = {
                                                'side': side,
                                                'initial_price': new_straddle_value,
                                                'delta': new_delta,
                                                'quantity': straddle_quantity,
                                                'trade_date': current_date_period,
                                                'maturity': expiry_date,
                                                'tau': tau_expiry,
                                                'strike': strike,
                                                'vol': vol,
                                                'initial_vol': vol,  # sigma_0 for P&L decomposition
                                                'risk_free': rf_T,
                                                'div_yield': div_yield,
                                                'put_price': put,
                                                'call_price': call,
                                                'current_gamma': new_gamma,
                                                'current_vega': new_vega,
                                                'current_volga': new_volga,
                                                'current_vol': vol,
                                                'current_tau': tau_expiry,
                                                'Gamma_star_prev': None,  # for dGamma term
                                                'F_prev': 0.0,  # for Vega term
                                                'last_price': None,
                                                'last_pnl': None,
                                                'closed_quantity': None
                                            }

            # ===================================================================
            # P&L DECOMPOSITION: Calculate the 4 terms according to the equation
            # ===================================================================
            if i > 0:  # Need at least one previous period
                dt = 1.0 / 252.0  # Trading days per year
                r_t = (current_price / prev_price) - 1.0  # Return
                
                # Calculate terms for each active position and aggregate
                gamma_term_increment = 0.0
                vega_term_increment = 0.0
                dgamma_term_increment = 0.0
                residual_term_increment = 0.0
                
                for key, pos in positions.items():
                    if pos['quantity'] == 0 or 'initial_vol' not in pos:
                        continue
                    
                    side = pos['side']  # -1 for short straddle
                    sigma_0 = pos['initial_vol']
                    sigma_t = pos.get('current_vol', sigma_0)
                    
                    # Get greeks (already calculated above)
                    gamma = pos.get('current_gamma', 0.0)
                    vega = pos.get('current_vega', 0.0)
                    volga = pos.get('current_volga', 0.0)
                    tau = pos.get('current_tau', 0.0)
                    
                    if tau <= 0:
                        continue
                    
                    # Calculate Gamma* = e^(-r*tau) * S^2 * Gamma
                    rf = pos.get('risk_free', 0.0)
                    Gamma_star = np.exp(-rf * tau) * (current_price ** 2) * gamma
                    
                    # 1. GAMMA TERM: 0.5 * Gamma* * ((r_t^2 / dt) - sigma_0^2) * dt
                    gamma_inc = side * 0.5 * Gamma_star * ((r_t**2 / dt) - sigma_0**2) * dt
                    gamma_term_increment += gamma_inc
                    
                    # 2. VEGA TERM: Uses F_t that "nets" to ~0 over life
                    # F_t = ((sigma_t + sigma_0) / (2 * sigma_t)) * vega * (sigma_t - sigma_0)
                    if sigma_t > 1e-8:
                        F_t = ((sigma_t + sigma_0) / (2.0 * sigma_t)) * vega * (sigma_t - sigma_0)
                    else:
                        F_t = 0.0
                    F_prev = pos.get('F_prev', 0.0)
                    vega_inc = side * (F_t - F_prev)
                    vega_term_increment += vega_inc
                    pos['F_prev'] = F_t  # Update for next period
                    
                    # 3. dGAMMA TERM: -(tau/2) * (sigma_t^2 - sigma_0^2) * dGamma*
                    Gamma_star_prev = pos.get('Gamma_star_prev', None)
                    if Gamma_star_prev is not None:
                        dGamma_star = Gamma_star - Gamma_star_prev
                        dgamma_inc = side * (- (tau / 2.0) * (sigma_t**2 - sigma_0**2) * dGamma_star)
                        dgamma_term_increment += dgamma_inc
                    pos['Gamma_star_prev'] = Gamma_star  # Update for next period
                    
                    # 4. RESIDUAL DRIFT TERM: -0.5 * e^(-r*u) * (vega/sigma) * volga * (d_sigma)^2
                    # Approximate d_sigma as the change from last period (would need more history for exact calculation)
                    # For now, use a simplified version based on current volga
                    u = (current_date_period - pos['trade_date']).days / 365.0
                    # Estimate d_sigma from position start (rough approximation)
                    d_sigma = sigma_t - sigma_0
                    if abs(sigma_t) > 1e-8:
                        res_inc = side * (-0.5 * np.exp(-rf * u) * (vega / sigma_t) * volga * (d_sigma ** 2))
                        residual_term_increment += res_inc
                
                # Update cumulative terms
                gamma_term_cum += gamma_term_increment
                vega_term_cum += vega_term_increment
                dgamma_term_cum += dgamma_term_increment
                residual_term_cum += residual_term_increment
            
            # Aggiorna hedge stock position
            stock_position_before = stock_position
            period_stock_pnl = stock_position_before * (current_price - prev_price)  # guadagno/perdita del periodo
            cumulative_stock_pnl += period_stock_pnl  # accumula il PnL dello stock hedging
            
            # Portfolio value = capitale + realized pnl + unrealized pnl opzioni + cumulative stock pnl
            portfolio_value = self.initial_capital + realized_pnl + straddle_pnl + cumulative_stock_pnl

            # Ribilancia: per delta-neutrality, compri -straddle_delta azioni
            # (se straddle venduto ha delta negativo, compri azioni positive)
            stock_position = -straddle_delta  # numero di azioni

            results.append({
                    'Date': current_date_period,
                    'Index Price': current_price,
                    'Index Return': performance - 1,
                    'Portfolio_Value': portfolio_value,
                    'Stock Position (shares)': stock_position,
                    'Stock Position Value': stock_position * current_price,
                    'Stock PnL (Period)': period_stock_pnl,
                    'Stock PnL (Cumulative)': cumulative_stock_pnl,
                    'Straddle Value': straddle_value,
                    'Straddle PnL (Unrealized)': straddle_pnl,
                    'Straddle PnL (Realized)': realized_pnl,
                    'Straddle PnL (Total)': realized_pnl + straddle_pnl,
                    'Straddle Delta': straddle_delta,
                    'Straddle Gamma': straddle_gamma,
                    'Straddle Vega': straddle_vega,
                    'Straddle Volga': straddle_volga,
                    'Active Positions': active_positions,
                    'Can Open New Positions': can_open_new_positions,
                    'Total Delta': straddle_delta + stock_position,  # dovrebbe essere ~0
                    # P&L Decomposition terms (cumulative)
                    'Gamma Term': gamma_term_cum,
                    'Vega Term': vega_term_cum,
                    'dGamma Term': dgamma_term_cum,
                    'Residual Term': residual_term_cum,
                    'First Order P&L': gamma_term_cum + vega_term_cum + dgamma_term_cum + residual_term_cum,
                })
        
        results = pd.DataFrame(results)
        positions = pd.DataFrame(positions)
        return results, positions


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


def option_price(
  valuation_date: date,
  maturity_date: date,
  spot_price: float,
  strike_price: float,
  volatility: float,
  risk_free_rate: float,
  dividend_rate: float,
  option_type: ql.Option,
  return_greeks: bool = False,
) -> tuple[float, float] | dict:
  """Price a European option using QuantLib's Black-Scholes-Merton analytic engine.

  Parameters are expressed in spot/strike units, annualized rates and vol, and Python `date`s.
  Returns a tuple (NPV, delta) where NPV is the option price and delta is the option delta.
  If return_greeks=True, returns a dict with NPV, delta, gamma, vega, and volga.
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
  
  if return_greeks:
    # Calculate volga (vomma) manually since QuantLib doesn't have it directly
    # Volga = vega * d1 * d2 / sigma
    # Using bump and revalue method for robustness
    vega = european_option.vega() / 100.0  # QuantLib returns vega per 1% change
    
    # Calculate volga using finite differences (bump vol by 1%)
    vol_bump = 0.01
    flat_vol_ts_up = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(valuation_date, calendar, volatility + vol_bump, day_count)
    )
    bsm_process_up = ql.BlackScholesMertonProcess(spot_handle, dividend_yield, flat_ts, flat_vol_ts_up)
    european_option_up = ql.VanillaOption(payoff, exercise)
    european_option_up.setPricingEngine(ql.AnalyticEuropeanEngine(bsm_process_up))
    vega_up = european_option_up.vega() / 100.0
    
    flat_vol_ts_down = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(valuation_date, calendar, max(volatility - vol_bump, 0.01), day_count)
    )
    bsm_process_down = ql.BlackScholesMertonProcess(spot_handle, dividend_yield, flat_ts, flat_vol_ts_down)
    european_option_down = ql.VanillaOption(payoff, exercise)
    european_option_down.setPricingEngine(ql.AnalyticEuropeanEngine(bsm_process_down))
    vega_down = european_option_down.vega() / 100.0
    
    # Volga = d(Vega)/d(sigma)
    volga = (vega_up - vega_down) / (2 * vol_bump)
    
    return {
      'npv': european_option.NPV(),
      'delta': european_option.delta(),
      'gamma': european_option.gamma(),
      'vega': vega,
      'volga': volga,
    }
  
  return european_option.NPV(), european_option.delta()

