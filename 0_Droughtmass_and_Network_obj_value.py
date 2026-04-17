import pypsa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Optional, Dict, List, Tuple
from contextlib import redirect_stdout # A clean way to redirect stdout
from datetime import datetime


factor_change_dunkleflaute="07"
sign="less"

# Configure logging (keep this)
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
pd.set_option('future.no_silent_downcasting', True)


###### All functions for the energy network ######

class ConfigNetwork:
    def __init__(self,
                 max_reservoir_volume,
                 inflows_time_series,
                 p_nominal, 
                 solar_production_time_series,
                 wind_production_time_series,
                 ror_production_time_series,
                 load_time_series,
                 swiss_energy_prices,
                 france_energy_prices,
                 austria_energy_prices,
                 italy_energy_prices,
                 germany_energy_prices,
                 debit,
                 network_snapshots):
        """
        Initialize the configuration for the national hydropower network.
        Parameters:
        - max_reservoir_volume: Maximum volume of the reservoir (m^3)
        - inflows_time_series: Time series of inflows to the reservoir (m^3/s)
        - p_nominal: Tuple of nominal power for turbine and pump (MW)
        - solar_production_time_series: Time series of solar production (MW)
        - wind_production_time_series: Time series of wind production (MW)
        - nuclear_production_time_series: Time series of nuclear production (MW)
        - ror_production_time_series: Time series of run-of-river production (MW)
        - load_time_series: Time series of load (electricity demand) (MW)
        """
        self.max_reservoir_volume = max_reservoir_volume
        self.inflows_time_series = inflows_time_series
        self.p_nominal = p_nominal
        self.solar_production_time_series = solar_production_time_series
        self.wind_production_time_series = wind_production_time_series
        self.ror_production_time_series = ror_production_time_series
        self.load_time_series = load_time_series
        self.swiss_energy_prices = swiss_energy_prices
        self.france_energy_prices = france_energy_prices
        self.austria_energy_prices = austria_energy_prices
        self.italy_energy_prices = italy_energy_prices
        self.germany_energy_prices = germany_energy_prices
        self.debit = debit
        self.network_snapshots = network_snapshots
        self.network = self.create_basic_electricity_network()
      
    def create_buses(self):
        """  
        Create buses for the PyPSA network and define their carriers.
        """
        network = pypsa.Network()
        network.set_snapshots(self.network_snapshots)

        # Add buses with explicit carrier definition
        network.add("Bus", "bus_national", carrier="DC")
        network.add("Bus", "bus_high_res")
        network.add("Bus", "bus_low_res")
        network.add("Bus", "bus_france", carrier="DC")
        network.add("Bus", "bus_austria", carrier="DC") 
        network.add("Bus", "bus_italy", carrier="DC")
        network.add("Bus", "bus_germany", carrier="DC")
    
        return network

    def add_national_hydropower_plant(self, network,
                                      max_reservoir_volume,
                                      inflows_time_series,
                                      p_nominal, debit):
        """
        Add a national hydropower plant to the network
        """
        # Unpack p_nominal
        p_nominal_turbine, p_nominal_pump = p_nominal
        debit_turbine, debit_pump = debit

        # Add a national hydropower plant
        # Added carrier for consistency warnings
        network.add("Store", "high_res", bus="bus_high_res", e_nom=max_reservoir_volume, e_initial=max_reservoir_volume * 0.5,e_cyclic=True, e_min_pu = 0.1, e_max_pu = 1) # e in m^3
        network.add("Store", "low_res", bus="bus_low_res", e_nom=1e15, e_initial=1e15 * 0.5) # e in m^3

        # Turbine Link: Electricity generated from water flow (p_set negative for generation)
        # p_nom is fixed, no p_nom_extendable
        network.add("Link", "Turbine", bus0="bus_national", bus1="bus_high_res", bus2="bus_low_res",
                    p_nom=p_nominal_turbine, # Use the fixed nominal power
                    efficiency=debit_turbine/p_nominal_turbine, # Water from high_res per MW
                    efficiency2=-debit_turbine/p_nominal_turbine, # Water to low_res per MW (negative for output)
                    marginal_cost=1,
                    p_min_pu=-1, p_max_pu=0)
                    #ramp_limit_up = 0.1,
                    #ramp_limit_down = 0.1) 


        network.add("Link", "Pump", bus0="bus_national", bus1="bus_low_res", bus2="bus_high_res",
                    p_nom=p_nominal_pump, # Use fixed nominal power
                    efficiency=-debit_pump/p_nominal_pump, # Water from low_res per MW
                    efficiency2=debit_pump/p_nominal_pump, # Water to high_res per MW (positive for input)
                    marginal_cost=1,
                    p_min_pu=0, p_max_pu=1)
                    #ramp_limit_up = 0.1,
                    #ramp_limit_down = 0.1) 

        network.add("Generator", "Inflows", bus="bus_high_res", p_set=inflows_time_series,
                    p_nom_extendable=True, p_nom = 0,
                    p_max_pu=1, p_min_pu=-1) # Added carrier

        return network

    def add_non_dispatchable_generators(self, network,
                                          solar_production_time_series,
                                          wind_production_time_series,
                                          ror_production_time_series):
        """
        Add existing non-dispatchable generators to the network
        For these, p_nom_extendable=True with p_nom=0 is generally fine when p_set is provided.
        PyPSA will set p_nom to be at least the max of p_set.
        Added carriers to prevent consistency warnings.
        """
        network.add("Generator", "Solar", bus="bus_national", p_set=solar_production_time_series,
                    p_nom_extendable=True, p_nom=0, p_max_pu=1, p_min_pu=0, carrier="DC")
        network.add("Generator", "Wind", bus="bus_national", p_set=wind_production_time_series,
                    p_nom_extendable=True, p_nom=0, p_max_pu=1, p_min_pu=0, carrier="DC")
        network.add("Generator", "ROR", bus="bus_national", p_set=ror_production_time_series,
                    p_nom_extendable=True, p_nom=0, p_max_pu=1, p_min_pu=0, carrier="DC")
        return network

    def add_import_export_links(self, network, swiss_energy_prices, france_energy_prices, austria_energy_prices, italy_energy_prices,germany_energy_prices):
        """
        Add import/export functionality using Generators and Lines.
        Imports are modeled as Generators connected to the national bus, with positive marginal cost.
        Exports are modeled as Generators connected to the national bus, with negative marginal cost (revenue).
        Lines represent the physical interconnections between countries.
        """
     
        # Import from France (Generator at bus_national)
        network.add("Generator", "Import_FR", bus="bus_france",
                    p_nom_extendable=False, p_nom=150000/4, # Max import capacity from France 
                    marginal_cost=france_energy_prices, # Cost of importing from France 
                    p_min_pu=0, p_max_pu=1, carrier="DC") # Can only "generate" (import) 

        # Export to France 
        network.add("Generator", "Export_FR", bus="bus_france",
                    p_nom_extendable=False, p_nom=150000/4, # Max export capacity to France 
                    marginal_cost=swiss_energy_prices, # Revenue from exporting (negative cost) 
                    p_min_pu=-1, p_max_pu=0, carrier="DC") # Can only "generate" (export) 
 
        # Line connecting National bus to France bus
        network.add("Line", "CH_to_FR_Line", bus0="bus_national", bus1="bus_france",
                    p_nom=150000/4, 
                    x=0.0001, r=0.0001, 
                    carrier="DC")
        network.add("Line", "FR_to_CH_Line", bus0="bus_france", bus1="bus_national",
                    p_nom=150000/4, 
                    x=0.0001, r=0.0001, 
                    carrier="DC")
        
        network.add("Generator", "Import_AT", bus="bus_austria",
                    p_nom_extendable=False, p_nom=150000/4, # Max import capacity from Austria 
                    marginal_cost=austria_energy_prices, # Cost of importing from Austria 
                    p_min_pu=0, p_max_pu=1, carrier="DC") # Can only "generate" (import)         
        network.add("Generator", "Export_AT", bus="bus_austria",
                    p_nom_extendable=False, p_nom=150000/4, # Max export capacity to Austria [cite: 24, 25]
                    marginal_cost=swiss_energy_prices, # Revenue from exporting (negative cost) [cite: 24, 25]
                    p_min_pu=-1, p_max_pu=0, carrier="DC") # Can only "generate" (export) [cite: 24, 25]

        # Line connecting National bus to Austria bus
        network.add("Line", "CH_to_AT_Line", bus0="bus_national", bus1="bus_austria",
                    p_nom=150000/4, 
                    x=0.0001, r=0.0001,
                    carrier="DC")
        network.add("Line", "AT_to_CH_line", bus0="bus_austria", bus1="bus_national",
                    p_nom = 150000/4, x=0.0001, r=0.0001, carrier="DC")

        # Import from Italy (Generator at bus_national)
        network.add("Generator", "Import_IT", bus="bus_italy",
                    p_nom_extendable=False, p_nom=150000/4, # Max import capacity from Italy [cite: 25, 26]
                    marginal_cost=italy_energy_prices, # Cost of importing from Italy [cite: 26]
                    p_min_pu=0, p_max_pu=1, carrier="DC") # Can only "generate" (import) [cite: 26, 27]

        # Export to Italy (Generator at bus_national, negative marginal cost for revenue)
        network.add("Generator", "Export_IT", bus="bus_italy",
                    p_nom_extendable=False, p_nom=150000/4, # Max export capacity to Italy [cite: 27]
                    marginal_cost=swiss_energy_prices, # Revenue from exporting (negative cost) [cite: 27]
                    p_min_pu=-1, p_max_pu=0, carrier="DC") # Can only "generate" (export) [cite: 27]

        # Line connecting National bus to Italy bus
        network.add("Line", "CH_to_IT_Line", bus0="bus_national", bus1="bus_italy",
                    p_nom=150000/4, # Transmission capacity
                    x=0.0001, r=0.0001,
                    carrier="DC")
        network.add("Line", "IT_to_CH_line", bus0="bus_italy", bus1="bus_national",
                    p_nom = 150000/4, x=0.0001, r=0.001, carrier="DC")
        
        network.add("Line", "CH_to_DE_Line", bus0="bus_national", bus1="bus_germany",
                    p_nom=150000/4, # Transmission capacity to Germany
                    x=0.0001, r=0.0001, carrier="DC")  
        network.add("Line", "DE_to_CH_line", bus0="bus_germany", bus1="bus_national",
                    p_nom = 150000/4, x=0.0001, r=0.001, carrier="DC")
        
        network.add("Generator", "Import_DE", bus="bus_national",
                    p_nom_extendable=False, p_nom=150000/4, # Max import capacity from Germany
                    marginal_cost=germany_energy_prices, # Cost of importing from Germany (assumed negligible)
                    p_min_pu=0, p_max_pu=1, carrier="DC")
        network.add("Generator", "Export_DE", bus="bus_national",
                    p_nom_extendable=False, p_nom=150000/4, # Max export capacity to Germany
                    marginal_cost=swiss_energy_prices, # Revenue from exporting (negative cost)
                    p_min_pu=-1, p_max_pu=0, carrier="DC")
        
        return network
    
    def add_load(self, network, load_time_series):
        """
        Add load (electricity demand) to the network
        For loads, p_nom_extendable=True with p_nom=0 is also generally fine when p_set is provided.
        PyPSA will set p_nom to be at least the max of abs(p_set).
        Added carrier to prevent consistency warnings.
        """
        network.add("Load", "Load", bus="bus_national", p_set=load_time_series)
        return network
        

    def create_basic_electricity_network(self):
        """
        Create a basic electricity network with a national hydropower plant,
        non-dispatchable generators, and load.
        """
        # Create a new PyPSA network
        network = self.create_buses()

        # Add national hydropower plant
        network = self.add_national_hydropower_plant(network,
                                                      self.max_reservoir_volume,
                                                      self.inflows_time_series,
                                                      self.p_nominal,
                                                      self.debit)

        # Add non-dispatchable generators
        network = self.add_non_dispatchable_generators(network,
                                                        self.solar_production_time_series,
                                                        self.wind_production_time_series,
                                                        self.ror_production_time_series)

        # Add load
        network = self.add_load(network, self.load_time_series)

        # Add import/export generator
        network = self.add_import_export_links(network, 
                                               self.swiss_energy_prices,
                                               self.france_energy_prices,
                                               self.austria_energy_prices,
                                               self.italy_energy_prices,
                                               self.germany_energy_prices)
        
        return network

    def add_solar_panels(self, time_series_production):
        """
        Add solar panels to the network - Note: This method is not called in your __main__
        """
        self.network.add("Generator", "Solar", bus="bus_national", p_set=time_series_production, p_nom_extendable=True, p_nom=0, p_max_pu=1, p_min_pu=0, carrier="electricity")

    def add_wind_turbines(self, time_series_production):
        """
        Add wind turbines to the network - Note: This method is not called in your __main__
        """
        self.network.add("Generator", "Wind", bus="bus_national", p_set=time_series_production, p_nom_extendable=True, p_nom=0, p_max_pu=1, p_min_pu=0, carrier="electricity")

    def optimize_powerflow(self):
        """
        Optimize the power flow in the network
        """
        print("\n--- Starting PyPSA Optimization ---")
        self.objective_value = None # Reset objective value before each run
        try:
            solver_output_file = "gurobi_output.log"
            solver_options = {}
            print(f"Redirecting gurobi solver output to: {solver_output_file}")

            with open(solver_output_file, 'w') as f:
                with redirect_stdout(f):
                    self.network.optimize(solver_name='gurobi', solver_options=solver_options)
                    
                    try:
                        self.objective_value = self.network.objective
                        print(f"Optimization Objective Value: {self.objective_value}")
                    except AttributeError as e:
                        print(f"Error accessing objective: {e}. This means PyPSA did not set it.")

            print("\n-- End Optimization --\n")

        except Exception as e:
                print(f"\n--- An unhandled error occurred during optimization: {e} ---")



def add_national_hydropower_plant(self, network,
                                    max_reservoir_volume,
                                    inflows_time_series,
                                    p_nominal, debit):
      """
      Add a national hydropower plant to the network
      """
      # Unpack p_nominal
      p_nominal_turbine, p_nominal_pump = p_nominal
      debit_turbine, debit_pump = debit

      # Add a national hydropower plant
      # Added carrier for consistency warnings
      network.add("Store", "high_res", bus="bus_high_res", e_nom=max_reservoir_volume, e_initial=max_reservoir_volume * 0.5,e_cyclic=True,
                  e_min_pu = 0.1, e_max_pu = 1) # e in m^3
      network.add("Store", "low_res", bus="bus_low_res", e_nom=1e15, e_initial=1e15 * 0.5) # e in m^3

      # Turbine Link: Electricity generated from water flow (p_set negative for generation)
      # p_nom is fixed, no p_nom_extendable
      network.add("Link", "Turbine", bus0="bus_national", bus1="bus_high_res", bus2="bus_low_res",
                  p_nom=p_nominal_turbine, # Use the fixed nominal power
                  efficiency=debit_turbine/p_nominal_turbine, # Water from high_res per MW
                  efficiency2=-debit_turbine/p_nominal_turbine, # Water to low_res per MW (negative for output)
                  marginal_cost=1,
                  p_min_pu=-1, p_max_pu=0)
                  #ramp_limit_up = 0.1,
                  #ramp_limit_down = 0.1) 


      network.add("Link", "Pump", bus0="bus_national", bus1="bus_low_res", bus2="bus_high_res",
                  p_nom=p_nominal_pump, # Use fixed nominal power
                  efficiency=-debit_pump/p_nominal_pump, # Water from low_res per MW
                  efficiency2=debit_pump/p_nominal_pump, # Water to high_res per MW (positive for input)
                  marginal_cost=1,
                  p_min_pu=0, p_max_pu=1)
                  #ramp_limit_up = 0.1,
                  #ramp_limit_down = 0.1) 

      network.add("Generator", "Inflows", bus="bus_high_res", p_set=inflows_time_series,
                  p_nom_extendable=True, p_nom = 0,
                  p_max_pu=1, p_min_pu=-1) # Added carrier

      return network


def compute_maximum_volume(res_features):
    """
    Sum up the maximum reservoir levels of every reservoir
    """
    #remove reservoirs with infinite volume (rivers)
    res_features = res_features[res_features['Vol'] < 1e10]
    max_volume = res_features['Vol'].sum() #m^3
    return max_volume



def compute_debit(plant_features):
    """
    Compute the debit of each turbine and pump
    """
    # Compute debit for turbines and pumps
    debit_turbine = plant_features['Q_T'].sum()* 3600   # m^3/h
    debit_pump = plant_features['Q_P'].sum() * 3600  # m^3/h
    return debit_turbine, debit_pump


def aggregate_inflows(inflows_time_series):
    """
    Aggregate inflows time series to a single time series
    """
    # Sum the inflows across all reservoirs for each time step
    print(inflows_time_series.head())  # Print for debugging
    aggregated_inflows = inflows_time_series.sum(axis = 1)
    return aggregated_inflows


def compute_p_nominal(plant_features):
    """
    Compute the total nominal power (MW) of the hydropower plant
    """
    # Sum the nominal power of all turbines and pumps
    p_nominal_turbine = plant_features['Pow_T'].sum()
    p_nominal_pump = plant_features['Pow_P'].sum()
    return p_nominal_turbine, p_nominal_pump



###### Drought Analyzer ######

class DroughtAnalyzer:
    def __init__(self, production_timeseries: pd.Series, demand_timeseries: pd.Series,
                 sma_window_sizes: list[int], thresholds: list[float],
                 start_date: pd.Timestamp, end_date: pd.Timestamp):
        """
        Initializes the DroughtAnalyzer with time series data and analysis parameters.

        Args:
            production_timeseries (pd.Series): The input supply time series data (e.g., actual production in MW).
                                               Index must be datetime.
            demand_timeseries (pd.Series): The demand time series data (e.g., in MW).
                                           Must have the same index and frequency as production_timeseries.
            sma_window_sizes (list[int]): A list of SMA window sizes in hours (e.g., [24*7, 24*30]).
            thresholds (list[float]): A list of threshold values to iterate over for each window.
            start_date (pd.Timestamp, optional): The start date for analysis. Defaults to None.
            end_date (pd.Timestamp, optional): The end date for analysis. Defaults to None.
        """
        self.production_timeseries = production_timeseries
        self.demand_timeseries = demand_timeseries
        self.sma_window_sizes = sma_window_sizes
        self.thresholds = thresholds
        self.start_date = start_date
        self.end_date = end_date

        self.all_results_by_window:Optional[Dict[int, Dict[float, List[Dict]]]] = None
        self.processed_production_timeseries: Optional[pd.Series] = None
        self.processed_demand_timeseries: Optional[pd.Series] = None
        self.drought_severity_metric: Optional[float] = None # This will be the total cumulative drought mass
        self.largest_drought_mass_increment: Optional[float] = None # This will be the most severe event score
        self.return_periods: Optional[Dict[int, Dict[float, Dict[str, float]]]] = None
        self.drought_length_statistics: Optional[Dict[str, float]] = None
        self.average_drought_length: Optional[float] = None # Derived from drought_length_statistics

    def _preprocess_data(self, time_series_data: pd.Series, demand_series: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Helper function to preprocess time series data (datetime indexing, timezone, duplicates).
        Ensures consistency and handles potential issues in input data.
        """
        ts_processed = time_series_data.copy()
        demand_processed = demand_series.copy()
    

        # Drop duplicate indices, keeping the first occurrence
        if not ts_processed.index.is_unique:
            print("Warning: Duplicate timestamps found in production_timeseries. Dropping duplicates (keeping first).")
            ts_processed = ts_processed[~ts_processed.index.duplicated(keep='first')]
        
        if not demand_processed.index.is_unique:
            print("Warning: Duplicate timestamps found in demand_timeseries. Dropping duplicates (keeping first).")
            demand_processed = demand_processed[~demand_processed.index.duplicated(keep='first')]
        
        return ts_processed, demand_processed


    def identify_mbt_droughts(self, time_series_data: pd.Series, sma_window_size: int, threshold_run: float) -> list:
        """
        Identifies Mean Below Threshold (MBT) renewable energy droughts using a robust
        groupby-based approach for finding consecutive periods.

        This function is adapted from the 'Drought_mass.txt' content.
        """
        ts_working = time_series_data.copy()
        actual_start = self.start_date if self.start_date is not None else ts_working.index.min()
        actual_end = self.end_date if self.end_date is not None else ts_working.index.max()

        # Reindex to a continuous hourly frequency and interpolate missing values
        idx = pd.date_range(actual_start, actual_end, freq='h')
        ts_working.index = pd.to_datetime(ts_working.index, format='mixed', errors='raise')# <--- Changed here
        ts_final = ts_working.reindex(idx).fillna(method='bfill').fillna(method='ffill').interpolate(method='linear').round(2) # <--- Changed here

        # Calculate Simple Moving Average (SMA)
        sma_array = ts_final.rolling(window=sma_window_size).mean() ### <- changed here

        # If SMA cannot be calculated (e.g., window too large for data), return empty list
        if np.array(sma_array==0).all():
            # print(f"Warning: SMA array for window size {sma_window_size}h is entirely NaN. Check data range and window size.")
            return []

        # Get the valid part of SMA (after initial NaNs from rolling window)
        sma_valid = sma_array.loc[pd.to_datetime(sma_array.first_valid_index()):] # <--- Changed here

        # Create the boolean series: True where SMA is below the threshold
        is_below_threshold = (sma_valid < threshold_run)

        drought_events_list = []

        
        # Use cumsum to assign a unique ID to each block of consecutive True values
        group_ids = (~is_below_threshold).cumsum()
        drought_blocks = is_below_threshold.loc[is_below_threshold].groupby(group_ids.loc[is_below_threshold])

        # Iterate through each identified drought block
        for group_id, block_series in drought_blocks:
            start_time = block_series.index.min()
            end_time = block_series.index.max()

            # Ensure the period is valid and has at least one data point
            if end_time >= start_time:
                event_sma_slice = sma_valid.loc[pd.to_datetime(start_time):pd.to_datetime(end_time)] # <--- Changed here

                if event_sma_slice.empty:
                    continue

                # Calculate Drought Characteristics
                duration_hours = (end_time - start_time).total_seconds() / 3600
                deficit = (threshold_run - event_sma_slice).sum()
                deficit_std = (threshold_run - event_sma_slice).std() if len(event_sma_slice) > 1 else np.nan
                min_sma = event_sma_slice.min()
                event_year = start_time.year

                drought_events_list.append({
                    'start': start_time,
                    'end': end_time,
                    'duration_hours': duration_hours,
                    'deficit': deficit,
                    'deficit_std': deficit_std,
                    'min_sma': min_sma,
                    'event_year': event_year
                })
        return drought_events_list

    def iterate_drought_analysis_over_thresholds(self, time_series_data: pd.Series,
                                                   sma_window_size: int,
                                                   thresholds: list[float]) -> dict:
        """
        Iterates the identify_mbt_droughts function over a list of specified thresholds.
        This function is adapted from the 'Drought_mass.txt' content.
        """
        results_by_threshold = {}
        for threshold in thresholds:
            drought_events = self.identify_mbt_droughts(
                time_series_data=time_series_data,
                sma_window_size=sma_window_size,
                threshold_run=threshold
            )
            results_by_threshold[threshold] = drought_events
        return results_by_threshold

    def analyze_droughts_across_windows(self):
        """
        Orchestrates drought analysis across multiple SMA time averaging windows and thresholds.
        Populates self.all_results_by_window, self.processed_production_timeseries, self.processed_demand_timeseries.
        This function is adapted from the 'Drought_mass.txt' content.
        """
        # Preprocess data once at the beginning of the analysis
        self.processed_production_timeseries, self.processed_demand_timeseries = self._preprocess_data(
            self.production_timeseries, self.demand_timeseries
        )

        all_results_by_window = {}
        for window_size in self.sma_window_sizes:
            # Skip if window size is larger than the available data
            if window_size > len(self.processed_production_timeseries):
                print(f"  Warning: SMA window size {window_size}h is larger than data length ({len(self.processed_production_timeseries)}h). Skipping.")
                continue

            results_for_this_window = self.iterate_drought_analysis_over_thresholds(
                time_series_data=self.processed_production_timeseries,
                sma_window_size=window_size,
                thresholds=self.thresholds
            )
            all_results_by_window[window_size] = results_for_this_window

        self.all_results_by_window = all_results_by_window

    def calculate_drought_severity_metric(self,
                                        overall_supply_weight: float = 0.8,
                                        overall_demand_weight: float = 0.2,
                                        weight_longest_drought_factor: float = 2.0,
                                        weight_most_severe_drought_factor: float = 2.0):
        """
        Calculates a single numerical drought severity metric (representing total cumulative drought mass)
        and identifies the largest drought mass increment (severity score of the most severe event).
        This function is adapted from the 'Drought_mass.txt' content.
        """
        if self.all_results_by_window is None:
            self.analyze_droughts_across_windows()

        # Internal fixed ratios for supply-side components (sum to 1.0)
        _SUPPLY_DURATION_RATIO = 0.7
        _SUPPLY_DEFICIT_RATIO = 0.3
        _SUPPLY_MIN_SMA_RATIO = 0

        # Internal fixed ratios for demand-side components (sum to 1.0)
        _DEMAND_AVG_DEMAND_RATIO = 0.6
        _DEMAND_MIN_SD_RATIO = 0.1
        _DEMAND_CUM_DEFICIT_RATIO = 0.3

        # Collect all events that can be processed (i.e., have valid supply/demand slices)
        # We will add the severity score directly to these event dictionaries.
        processable_events = []
        for window_size, results_by_threshold in self.all_results_by_window.items():
            for threshold, drought_list in results_by_threshold.items():
                for event in drought_list:
                    # Calculate Demand-Related Metrics for this specific drought event
                    self.processed_production_timeseries.index = pd.to_datetime(self.processed_production_timeseries.index, format='mixed', errors='raise')
                    self.processed_demand_timeseries.index = pd.to_datetime(self.processed_demand_timeseries.index, format='mixed', errors='raise')

                    drought_supply_slice = self.processed_production_timeseries.loc[event['start']:event['end']]
                    drought_demand_slice = self.processed_demand_timeseries.loc[event['start']:event['end']]

                    drought_demand_slice = drought_demand_slice.reindex(drought_supply_slice.index)

                    # Skip this event if data is insufficient for demand-side calculation
                    if drought_supply_slice.empty or drought_demand_slice.empty or drought_supply_slice.isnull().all().all() or drought_demand_slice.isnull().all().all():
                        continue 
                    
                    # Add identified window size and threshold directly to the event dictionary
                    event['identified_window_size'] = window_size
                    event['identified_threshold'] = threshold
                    event['avg_demand_during_event'] = drought_demand_slice.mean()
                    supply_minus_demand = drought_supply_slice - drought_demand_slice
                    event['min_supply_minus_demand_during_event'] = supply_minus_demand.min()
                    event['cumulative_supply_deficit_relative_to_demand'] = (
                        (drought_demand_slice - drought_supply_slice).clip(lower=0).sum()
                    )
                    
                    processable_events.append(event) # Add the original event (now updated)

        if not processable_events:
            self.drought_severity_metric = 0.0 # Total cumulative drought mass
            self.largest_drought_mass_increment = 0.0 # Largest drought mass increment
            return

        # Determine Normalization Maxima/Minima for all metrics across ALL processable events
        max_duration_overall = max(e['duration_hours'] for e in processable_events) if processable_events else 0
        max_deficit_overall = max(e['deficit'] for e in processable_events) if processable_events else 0
        min_sma_overall = min(e['min_sma'] for e in processable_events) if processable_events else 0
        
        max_avg_demand_overall = max(e['avg_demand_during_event'].values[0] for e in processable_events) if processable_events else 0                                                       # <-- Changed here
        min_supply_minus_demand_overall = min(e['min_supply_minus_demand_during_event'].values[0] for e in processable_events) if processable_events else 0                                 # <-- Changed here     
        max_cumulative_deficit_relative_to_demand_overall = max(e['cumulative_supply_deficit_relative_to_demand'].values[0] for e in processable_events) if processable_events else 0       # <-- Changed here

        max_possible_sma_ref = self.processed_production_timeseries.max() 
        if max_possible_sma_ref <= 0:
            max_possible_sma_ref = 1.0 

        all_threshold_values = sorted(list(set(e['identified_threshold'] for e in processable_events)))
        min_threshold_value = min(all_threshold_values) if all_threshold_values else 0
        max_threshold_value = max(all_threshold_values) if all_threshold_values else 0

        all_window_sizes = sorted(list(set(e['identified_window_size'] for e in processable_events)))
        min_window_size = min(all_window_sizes) if all_window_sizes else 0
        max_window_size = max(all_window_sizes) if all_window_sizes else 0

        # Helper function to calculate the weight for a given threshold.
        def get_threshold_weight(current_threshold, min_t, max_t):
            if max_t == min_t:
                return 1.0
            normalized_pos = (max_t - current_threshold) / (max_t - min_t)
            return 1.0 + normalized_pos # Weight ranges from 1.0 (highest threshold) to 2.0 (lowest threshold)

        # Helper function to calculate the weight for a given window size.
        def get_window_size_weight(current_window_size, min_w, max_w):
            if max_w == min_w:
                return 1.0
            normalized_pos = (current_window_size - min_w) / (max_w - min_w)
            return 1.0 + normalized_pos # Weight ranges from 1.0 (shortest window) to 2.0 (longest window)

        total_weighted_event_score = 0.0
        event_severities_for_ranking = []
        longest_drought_duration = 0.0

        for event in processable_events: # Iterate over the processable events
            # Track the longest drought duration for its contribution
            if event['duration_hours'] > longest_drought_duration:
                longest_drought_duration = event['duration_hours']

            # Normalize supply-side characteristics
            norm_duration = event['duration_hours'] / max_duration_overall if max_duration_overall > 0 else 0
            norm_deficit = event['deficit'] / max_deficit_overall if max_deficit_overall > 0 else 0
            norm_min_sma_severity = (max_possible_sma_ref - event['min_sma']) / max_possible_sma_ref
            norm_min_sma_severity = max(0, min(1, norm_min_sma_severity)) 

            # Calculate combined supply-side severity score for this event
            supply_severity_score = (
                norm_duration * _SUPPLY_DURATION_RATIO +
                norm_deficit * _SUPPLY_DEFICIT_RATIO +
                norm_min_sma_severity * _SUPPLY_MIN_SMA_RATIO
            )

            # Normalize demand-side characteristics
            norm_avg_demand = event['avg_demand_during_event'] / max_avg_demand_overall if max_avg_demand_overall > 0 else 0
            
            norm_min_supply_minus_demand_severity = 0
            if min_supply_minus_demand_overall < 0:
                norm_min_supply_minus_demand_severity = (
                    event['min_supply_minus_demand_during_event'] - 0
                ) / (min_supply_minus_demand_overall - 0)
                norm_min_supply_minus_demand_severity = 1 - norm_min_supply_minus_demand_severity
            norm_min_supply_minus_demand_severity = max(0, min(1, norm_min_supply_minus_demand_severity)) 

            norm_cumulative_deficit_relative_to_demand = (
                event['cumulative_supply_deficit_relative_to_demand'] / max_cumulative_deficit_relative_to_demand_overall
                if max_cumulative_deficit_relative_to_demand_overall > 0 else 0
            )

            # Calculate combined demand-side impact score for this event
            demand_impact_score = (
                norm_avg_demand * _DEMAND_AVG_DEMAND_RATIO +
                norm_min_supply_minus_demand_severity * _DEMAND_MIN_SD_RATIO +
                norm_cumulative_deficit_relative_to_demand * _DEMAND_CUM_DEFICIT_RATIO
            )
            
            # Combine supply and demand scores using the overall tunable weights
            event_base_score = (
                supply_severity_score * overall_supply_weight +
                demand_impact_score * overall_demand_weight
            )
            
            # Apply threshold weighting AND new window size weighting
            threshold_weight = get_threshold_weight(
                event['identified_threshold'], min_threshold_value, max_threshold_value
            )
            window_size_weight = get_window_size_weight(
                event['identified_window_size'], min_window_size, max_window_size
            )
            
            # Store the individual event severity score for plotting directly in the event dict
            event['event_severity_score_for_plot'] = event_base_score * threshold_weight * window_size_weight
            
            total_weighted_event_score += event['event_severity_score_for_plot']
            # Prepare data for ranking the "most severe" event
            supply_severity_for_ranking = (
                norm_deficit * _SUPPLY_DEFICIT_RATIO +
                norm_min_sma_severity * _SUPPLY_MIN_SMA_RATIO
            )
            demand_impact_for_ranking = (
                norm_cumulative_deficit_relative_to_demand * _DEMAND_CUM_DEFICIT_RATIO +
                norm_min_supply_minus_demand_severity * _DEMAND_MIN_SD_RATIO
            )
            event_severity_score = (
                supply_severity_for_ranking * overall_supply_weight +
                demand_impact_for_ranking * overall_demand_weight
            )
            event_severities_for_ranking.append(event_severity_score)

        # Incorporate Global Longest and Most Severe Drought Contributions
        longest_drought_contribution = longest_drought_duration * weight_longest_drought_factor
        
        most_severe_event_score_val = 0.0
        most_severe_contribution = 0.0
        if event_severities_for_ranking:
            most_severe_event_score_val = max(event_severities_for_ranking)
            most_severe_contribution = most_severe_event_score_val * weight_most_severe_drought_factor

        # Final Drought Severity Index (DSI) - This represents the "total cumulative drought mass"
        # The division by 3.0 is an arbitrary scaling factor to keep the DSI in a reasonable range.
        dsi = (total_weighted_event_score + longest_drought_contribution + most_severe_contribution) / 3.0
        
        self.drought_severity_metric = dsi
        self.largest_drought_mass_increment = most_severe_event_score_val # This is the "largest drought mass increment"
        
        return dsi
    
    def calculate_return_periods(self, analysis_period_years: Optional[float] = None) -> Dict[int, Dict[float, Dict[str, float]]]:
        """
        Calculates return periods for drought events at different thresholds and window sizes.
        Populates self.return_periods.
        This function is adapted from the 'return periods.txt' content.
        """
        if self.all_results_by_window is None:
            self.analyze_droughts_across_windows()

        # Calculate analysis period if not provided
        if analysis_period_years is None:
            if self.processed_production_timeseries.empty:
                raise ValueError("Cannot calculate analysis period from empty time series data")

#            time_range = self.processed_production_timeseries.index.max() - self.processed_production_timeseries.index.min()
            time_range = datetime.strptime(str(self.processed_production_timeseries.index.max()), '%Y-%m-%d %H:%M:%S')-datetime.strptime(str(self.processed_production_timeseries.index.min()), '%Y-%m-%d %H:%M:%S')
            analysis_period_years = time_range.total_seconds() / (365.25 * 24 * 3600)
            
            if analysis_period_years <= 0:
                raise ValueError("Analysis period must be positive")
        
        return_period_results = {}
        
        for window_size, results_by_threshold in self.all_results_by_window.items():
            return_period_results[window_size] = {}
            
            for threshold, drought_events in results_by_threshold.items():
                event_count = len(drought_events)
                
                # min_events_for_calculation hardcoded to 2, as per original function
                if event_count < 2:
                    # Not enough events for reliable return period calculation
                    return_period_results[window_size][threshold] = {
                        'return_period_years': np.nan,
                        'return_period_days': np.nan,
                        'event_count': event_count,
                        'events_per_year': event_count / analysis_period_years if analysis_period_years > 0 else 0,
                        'analysis_period_years': analysis_period_years,
                        'insufficient_data': True
                    }
                else:
                    # Calculate return period
                    events_per_year = event_count / analysis_period_years
                    return_period_years = 1 / events_per_year if events_per_year > 0 else np.inf
                    return_period_days = return_period_years * 365.25
                    
                    return_period_results[window_size][threshold] = {
                        'return_period_years': return_period_years,
                        'return_period_days': return_period_days,
                        'event_count': event_count,
                        'events_per_year': events_per_year,
                        'analysis_period_years': analysis_period_years,
                        'insufficient_data': False
                    }
        self.return_periods = return_period_results
        return return_period_results

    def extract_drought_length_statistics(self) -> dict:
        """
        Extracts comprehensive drought length statistics over all thresholds and windows.
        Populates self.drought_length_statistics and self.average_drought_length.
        This function is adapted from the 'Drought lengths.txt' content.
        """
        if self.all_results_by_window is None:
            self.analyze_droughts_across_windows()

        all_durations = []
        # Flatten the nested dictionary to collect all drought durations
        for window_size, results_by_threshold in self.all_results_by_window.items():
            for threshold, drought_list in results_by_threshold.items():
                for event in drought_list:
                    if 'duration_hours' in event:
                        all_durations.append(event['duration_hours'])

        if not all_durations:
            stats = {
                'average_duration': 0.0,
                'median_duration': 0.0,
                'min_duration': 0.0,
                'max_duration': 0.0,
                'std_duration': 0.0,
                'total_events': 0,
                'total_drought_time': 0.0
            }
        else:
            # Calculate statistics
            stats = {
                'average_duration': np.mean(all_durations),
                'median_duration': np.median(all_durations),
                'min_duration': np.min(all_durations),
                'max_duration': np.max(all_durations),
                'std_duration': np.std(all_durations),
                'total_events': len(all_durations),
                'total_drought_time': np.sum(all_durations)
            }
        self.drought_length_statistics = stats
        self.average_drought_length = stats['average_duration']
        return stats

    def get_drought_mass_time_series(self) -> List[Dict]:
        """
        Returns the raw identified drought events, which can be considered the "drought mass time series".
        Each event dictionary contains details like start, end, duration, deficit, etc.
        """
        if self.all_results_by_window is None:
            self.analyze_droughts_across_windows()
        
        all_events = []
        for window_size, results_by_threshold in self.all_results_by_window.items():
            for threshold, drought_list in results_by_threshold.items():
                for event in drought_list:
                    all_events.append(event)
        return all_events

    def run_all_analysis(self):
        """
        Runs all drought analysis calculations and populates the class attributes.
        Call this method to perform the full analysis after initializing the class.
        """
        self.analyze_droughts_across_windows()
        self.calculate_drought_severity_metric()
        self.calculate_return_periods()
        self.extract_drought_length_statistics()

    def plot_drought_results(self, return_period_window_size: int):
        """
        Generates plots for drought mass time series (accumulation and individual events)
        and return periods with respect to threshold.

        Args:
            return_period_window_size (int): The SMA window size to use for the return period plot.
        """
        if self.all_results_by_window is None:
            print("Please run run_all_analysis() first to generate drought results.")
            return
        
        # --- Plot 1: Drought Mass Accumulation Over Time ---
        all_events = self.get_drought_mass_time_series()
        
        # Filter for events that have the severity score for plotting
        events_with_scores = [event for event in all_events if 'event_severity_score_for_plot' in event]

        if not events_with_scores:
            print("No drought events with severity scores found to plot drought mass time series.")
        else:
            # Sort events by their end time to ensure correct accumulation
            all_events_sorted = sorted(events_with_scores, key=lambda x: x['end'])
            
            # Use the 'event_severity_score_for_plot' for accumulation
            event_times = [event['end'] for event in all_events_sorted]
            event_severity_scores = [event['event_severity_score_for_plot'] for event in all_events_sorted]
            
            cumulative_severity_score = np.cumsum(event_severity_scores)

            plt.figure(figsize=(12, 6))
            plt.plot(event_times, cumulative_severity_score, marker='o', linestyle='-', markersize=4, color='blue')
            plt.title('Cumulative Drought Severity Score Over Time (80% Solar, 20% Wind)')
            plt.xlabel('End Date of Drought Event')
            plt.ylabel('Cumulative Drought Severity Score (DSI)')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.show()

        # --- Plot 2: Individual Drought Severity Scores (Bar Graph) ---
        if not events_with_scores:
            # Message already printed above if no events
            pass
        else:
            # Using start time for x-axis for individual bars
            event_starts = [event['start'] for event in all_events_sorted]
            
            event_severity_scores = [event['event_severity_score_for_plot'] for event in all_events_sorted]
      
            plt.figure(figsize=(12, 6))
            # Adjust bar width dynamically based on typical time difference between events
            if len(event_starts) > 1:
                # Calculate average time difference in days for bar width, ensuring it's positive
                time_diffs_hours = [(event_starts[i+1] - event_starts[i]).total_seconds() / 3600 for i in range(len(event_starts)-1)]
                avg_time_diff_hours = np.mean([diff for diff in time_diffs_hours if diff > 0])
                if avg_time_diff_hours > 0:
                    bar_width_days = avg_time_diff_hours / 24.0 * 0.8 # 80% of average time difference in days
                else:
                    bar_width_days = 1.0 # Default to 1 day if all events are at the same timestamp or very close
            else:
                bar_width_days = 1.0 # Default to 1 day if only one event

            # Convert bar width from days to a format compatible with matplotlib's bar width (which is typically numeric)
            # For datetime x-axis, width is usually in "data units" which can be days.
            plt.bar(event_starts, event_severity_scores, width=pd.Timedelta(days=bar_width_days), color='salmon', edgecolor='black')
            plt.title('Individual Drought Severity Scores Over Time (20% Solar, 80% Wind)')
            plt.xlabel('Drought Start Date')
            plt.ylabel('Drought Severity Score (DSI Increment)')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()

        # --- Plot 3: Return Periods vs. Threshold ---
        if self.return_periods is None:
            print("Please run calculate_return_periods() first to generate return period data.")
            return

        if return_period_window_size not in self.return_periods:
            print(f"Return period data for window size {return_period_window_size}h not found.")
            print(f"Available window sizes: {list(self.return_periods.keys())}")
            return

        threshold_data = self.return_periods[return_period_window_size]
        
        thresholds = []
        return_periods = []
        event_counts = []
        
        for threshold, data in threshold_data.items():
            if not data['insufficient_data'] and np.isfinite(data['return_period_years']):
                thresholds.append(threshold)
                return_periods.append(data['return_period_years'])
                event_counts.append(data['event_count'])
        
        if not thresholds:
            print(f"No valid data points for plotting return period curves for window size {return_period_window_size}h.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Return Period vs Threshold
        ax1.plot(thresholds, return_periods, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Drought Threshold')
        ax1.set_ylabel('Return Period (years)')
        ax1.set_title(f'Drought Return Period Analysis\nWindow Size: {return_period_window_size}h')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log') # Log scale is often useful for return periods
        
        # Plot 2: Event Count vs Threshold
        ax2.bar(thresholds, event_counts, alpha=0.7, color='coral')
        ax2.set_xlabel('Drought Threshold')
        ax2.set_ylabel('Number of Events')
        ax2.set_title(f'Event Frequency\nWindow Size: {return_period_window_size}h')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()    




####### process time series #######
def rescale(wind_ts: pd.Series, pv_ts: pd.Series, wind_ratio: float):

    target_MWh = 25000000*5

    pv_ratio = 1.0 - wind_ratio
    target_wind_MWh = target_MWh * wind_ratio
    target_pv_MWh = target_MWh * pv_ratio

    original_wind_MWh = wind_ts.sum()
    original_pv_MWh = pv_ts.sum()

    scaling_factor_wind = target_wind_MWh / original_wind_MWh

    scaling_factor_pv = target_pv_MWh / original_pv_MWh

    adjusted_wind_ts = wind_ts * scaling_factor_wind
    adjusted_pv_ts = pv_ts * scaling_factor_pv

    return adjusted_wind_ts, adjusted_pv_ts


###### initialize the time series ######
network_snapshots = pd.date_range(start='2015-01-01 00:00:00', end='2020-01-01', freq='h')[0:(len(pd.date_range(start='2015-01-01', end='2020-01-01', freq='h'))-1)]
start_date = network_snapshots[0]
end_date = network_snapshots[-1]
results = []

#Analysis parameters:
solar_ratios = np.arange(0, 1, 0.05) #Wind ratio = 1 - solar ratio
sma_window_sizes = np.array([24*7, 24*30, 24*91, 24*365]) # 1 week, 1 month, 3 months, 1 year
threshold_percentages = np.array([0.6, 0.5, 0.4, 0.3, 0.2, 0.1])

#Load and process data:
#reservoir features
res_features = pd.read_feather("/Users/privoire/Documents/PostDocEPFL/Bernie/Data/Other_RE/HydroRes_data.ftr")
res_features = res_features.set_index('Bus') 
# The reservoirs as bus (bus in PYPSA)

#plant features
plant_features = pd.read_feather("/Users/privoire/Documents/PostDocEPFL/Bernie/Data/Other_RE/HydroPlant_data.ftr")
plant_features = plant_features.set_index('Bus')
#herre bus=electric bus, high res=reservoir above, low res=reservoir below

#inflows
inflows_time_series = pd.read_feather("/Users/privoire/Documents/PostDocEPFL/Bernie/Data/Other_RE/Inflow_hourly_2015-2019.ftr")

#Load and prepare synthetic data    
synthetic_data_noprice = pd.read_csv("/Users/privoire/Documents/PostDocEPFL/Bernie/Synthetic_data_dunkleflaute/Synthetic_wind_PV_demand_meteo4price_CH_meteo4price_neighbours_"+
                             factor_change_dunkleflaute+"times_"+sign+"_dunkelflautedays_withsythetic_date.csv")

solar_production_time_series_raw = synthetic_data_noprice[["synthetic_hours",'PV']].set_index("synthetic_hours")
solar_production_time_series_raw.index = pd.to_datetime(solar_production_time_series_raw.index, format='mixed', errors='raise') 
wind_production_time_series_raw = synthetic_data_noprice[["synthetic_hours",'Wind']].set_index("synthetic_hours")
wind_production_time_series_raw.index = pd.to_datetime(wind_production_time_series_raw.index, format='mixed', errors='raise')   
load_time_series = synthetic_data_noprice[["synthetic_hours",'Demand']].set_index("synthetic_hours")
load_time_series.index = pd.to_datetime(load_time_series.index, format='mixed', errors='raise') 

#Load and prepare synthetic data    
synthetic_data_price = pd.read_csv("/Users/privoire/Documents/PostDocEPFL/Bernie/Synthetic_data_dunkleflaute/Synthetic_prices/electricity_prices_"+
                             factor_change_dunkleflaute+".csv", index_col=0, parse_dates=True)

swiss_energy_prices = synthetic_data_price[["electricity_prices_CH"]]
france_energy_prices = synthetic_data_price[["electricity_prices_FR"]]
austria_energy_prices = synthetic_data_price[["electricity_prices_AT"]]
italy_energy_prices = synthetic_data_price[["electricity_prices_IT"]]
germany_energy_prices = synthetic_data_price[["electricity_prices_DE"]]

# non-dispatchable resources
nuclear_production_time_series_raw = 0 #pd.read_csv("C:/Users/berni/Downloads/Network_data/nuclear_production_time_series_hourly.csv", index_col=0, parse_dates=True).squeeze()

ror_production_time_series = pd.read_feather("/Users/privoire/Documents/PostDocEPFL/Bernie/Data/Other_RE/ROR_hourly_2015-2019.ftr")
# aggregated by electrical node


#MAIN FOR LOOP - Optimises PyPSA network and runs drought analysis for different solar ratios
for ratio in solar_ratios:
    print(f"\n--- Analyzing solar ratio: {ratio:.1f} ---")
    
    # Rescale the time series

    rescaled_wind_ts, rescaled_solar_ts = rescale(pv_ts=solar_production_time_series_raw,
                                                  wind_ts=wind_production_time_series_raw, wind_ratio=ratio)

    mean_production_total = rescaled_solar_ts.mean().PV + rescaled_wind_ts.mean().Wind # --> Was changed here

    thresholds = mean_production_total * threshold_percentages

    # Create and optimize the network with the rescaled data
    instance_network = ConfigNetwork(
        max_reservoir_volume=compute_maximum_volume(res_features),
        inflows_time_series=inflows_time_series.sum(axis=0),                        # --> Was changed here!!!!
        p_nominal=compute_p_nominal(plant_features),
        solar_production_time_series=rescaled_solar_ts.PV.values,                   # --> Was changed here
        wind_production_time_series=rescaled_wind_ts.Wind.values,                   # --> Was changed here
        ror_production_time_series=ror_production_time_series.sum(axis=0),          # --> Was changed here!!!! because demand was summed
        load_time_series=load_time_series.Demand.values,                            # --> Was changed here
        swiss_energy_prices=swiss_energy_prices.electricity_prices_CH.values,       # --> Was changed here
        france_energy_prices=france_energy_prices.electricity_prices_FR.values,     # --> Was changed here
        austria_energy_prices=austria_energy_prices.electricity_prices_AT.values,   # --> Was changed here
        italy_energy_prices=italy_energy_prices.electricity_prices_IT.values,       # --> Was changed here
        germany_energy_prices=germany_energy_prices.electricity_prices_DE.values,   # --> Was changed here
        debit=compute_debit(plant_features),
        network_snapshots = network_snapshots
    )
    instance_network.optimize_powerflow()

    # Run the drought analysis
    analyzer = DroughtAnalyzer(
        production_timeseries=rescaled_solar_ts.PV + rescaled_wind_ts.Wind,       # --> Was changed here
        demand_timeseries=load_time_series,
        sma_window_sizes=sma_window_sizes,
        thresholds=thresholds,
        start_date=start_date,
        end_date=end_date
    )
    analyzer.run_all_analysis()

    # Store the results in a dictionary'''
    results.append({
        'solar_ratio': ratio,
        'objective_value': instance_network.objective_value,
        'drought_mass': analyzer.drought_severity_metric.iloc[0]               #--> Was changed here to access the scalar value from the Series
    })

drought_lengths = analyzer.extract_drought_length_statistics()
print(drought_lengths)
# Save the results to a CSV file
print("\n--- Saving results to CSV file ---")
results_df = pd.DataFrame(results)
#name_file="sythetic"+factor_change_dunkleflaute+"x"+sign+".csv"
results_df.to_csv("synthetic"+factor_change_dunkleflaute+"x"+sign+".csv", index=False)
print("Results saved to ~/synthetic"+factor_change_dunkleflaute+"x"+sign+".csv")