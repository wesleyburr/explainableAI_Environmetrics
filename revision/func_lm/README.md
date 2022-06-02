Folder containing revised code and data for the functional linear model

stat_model_eda.R -- plots the US Corn Belt locations and the marginal correlations of the soil moisture data in the cornbelt with 3-month lagged SST

stat_model_all_May24_2022 -- conducts the remaining analysis including generating persistence and climatology fits, 
running the functional linear model, and obtaining model reliance metrics for the stat_model_all

maps_and_metrics.R -- generating metrics tables and plots from the fits. In particular, it takes such an output file and produces the model comparison metrics table, plots of true vs fitted, and maps of MSPE by locations.

To use maps_and_metrics.R for generating metrics for other methods, use the following steps:

1)	Create the output from your method as a long-format table/dataframe containing atleast these columns -- sm_loc_id (SM cornbelt location id),	Lon,	Lat,	date,	value (true SM value),	fit (SM predictions from the model).
2)	Store the fits as a csv file in a folder titled “outputs”, e.g., if you are using ANN, call the file “ANN_fits.csv”
3)	Create two folders – “metrics” (stores the model comp. metrics) and “figures” (stores the plots and maps)
4)	Load the file maps_and_metrics.R 
5)	Run metrics_gen("ANN")

modelreliance.Rdata -- saved R object containing the modelreliance metrics for the functional linear model (this is saved as obtaning the metrics can be time-consuming)
