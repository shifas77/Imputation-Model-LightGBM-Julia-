D:
cd Julia-1.7.2\bin

set JULIA_NUM_THREADS=4
SET JULIA_DEPOT_PATH=D:\JULIA_ASSETS\JULIA_PACKAGE_DIRECTORY

julia

using Pkg
using AutoMLPipeline
using Random
using CategoricalArrays
using DataFrames
using CSV
using StableRNGs
using MLJ
using CSVFiles
using Statistics



#Initializing
miss_class_all=[]
models_all=[]


print("Running Fine")





final_dup= DataFrame(CSVFiles.load("C:\\Users\\user\\Documents\\GOOGLE_CYCLIST2.csv"))
#final_dup=CSV.read("C:\\Users\\user\\Documents\\GOOGLE_CYCLIST2.csv", DataFrame)
print("Done Loading")

Tempo=[replace(final_dup[:,var],"" => missing) for var in 1:ncol(final_dup)]
Temp2=DataFrame(Tempo,:auto)
final_dup=rename(Temp2, names(final_dup))

missing_names=describe(final_dup,:nmissing).variable[findall(describe(final_dup,:nmissing).nmissing.>0)]

missing_index=[findall(ismissing.(final_dup[:,var])) for var in missing_names]

Labels=names(select(final_dup,[:start_station_name,:start_station_id,
:end_station_name,:end_station_id]))

To_be_conv=select(final_dup,Not(Labels))

To_be_conv.end_lat=replace(To_be_conv[:,:end_lat],missing => mean(skipmissing(To_be_conv.end_lat)))
To_be_conv.end_lng=replace(To_be_conv[:,:end_lng],missing => mean(skipmissing(To_be_conv.end_lng)))

To_be_conv=coerce(To_be_conv,Count => Multiclass)
To_be_conv=coerce(To_be_conv,Textual=> Multiclass)
To_be_conv=coerce(To_be_conv,Multiclass=>Continuous)
To_be_conv=coerce(To_be_conv,Union{Missing,Continuous}=> Multiclass)
To_be_conv=coerce(To_be_conv,Multiclass=>Continuous)

Hope=final_dup[:,Labels]
Hope_fin=hcat(Hope,To_be_conv)

final_dup=Hope_fin

dropmissing!(final_dup)
x=DataFrames.select(final_dup,Not([:start_station_name,:start_station_id,
:end_station_name,:end_station_id]))
y=DataFrames.select(final_dup,[:start_station_name,:start_station_id,
:end_station_name,:end_station_id])

perm = randperm(nrow(y))
rng=StableRNG(perm[1])
train,test=partition(eachindex(y[:,1]),0.0425, shuffle=true ,rng=rng)



for i in 1:ncol(y)
y_hat=categorical(y[:,i])
print("Done Coercing")
LGBMClassifier= @load LGBMClassifier
xgb_model  =LGBMClassifier(
    boosting = "gbdt",
    num_iterations = 15,
    learning_rate = 0.1,
    num_leaves = 32,
    max_depth = 6,
    tree_learner = "serial",
    histogram_pool_size = -1.0,
    min_data_in_leaf = 20,
    min_sum_hessian_in_leaf = 0.001,
    max_delta_step = 0.0,
    lambda_l1 = 0.0,
    lambda_l2 = 2,
    min_gain_to_split = 0.5,
    feature_fraction = 1.0,
    feature_fraction_bynode = 1.0,
    feature_fraction_seed = 2,
    bagging_fraction = 0.00325,
    pos_bagging_fraction = 1.0,
    neg_bagging_fraction = 1.0,
    bagging_freq = 0,
    bagging_seed = 3,
    early_stopping_round = 0,
    extra_trees = false,
    extra_seed = 6,
    max_bin = 255,
    bin_construct_sample_cnt = 200000,
    init_score = "",
    drop_rate = 0.1,
    max_drop = 50,
    skip_drop = 0.5,
    xgboost_dart_mode = false,
    uniform_drop = false,
    drop_seed = 4,
    top_rate = 0.2,
    other_rate = 0.1,
    min_data_per_group = 100,
    max_cat_threshold = 32,
    cat_l2 = 10.0,
    cat_smooth = 10.0,
    objective = "multiclass",
    categorical_feature = Int64[],
    data_random_seed = 1,
    is_sparse = true,
    is_unbalance = true,
    boost_from_average = true,
    scale_pos_weight = 1.0,
    use_missing = true,
    feature_pre_filter = true,
    metric =["multi_logloss"],
    metric_freq = 1,
    is_training_metric = false,
    ndcg_at = [1, 2, 3, 4, 5],
    num_machines = 1,
    num_threads = Sys.CPU_THREADS,
    local_listen_port = 12400,
    time_out = 120,
    machine_list_file = "",
    save_binary = false,
    device_type = "cpu",
    force_col_wise = true,
    force_row_wise = false,
    truncate_booster = true)
MLJ.schema(x)
mach=machine(xgb_model,x,Vector(y_hat))
fin_model=MLJ.fit!(mach,rows=train)
#Predicting
rng = MersenneTwister(1234)
num_rounds=5
row_value=[shuffle(rng,test)[1:100000] for var in 1:num_rounds]
final_miss_class=[misclassification_rate(mode.(MLJ.predict(fin_model,rows=row_value[i])), y_hat[row_value[i]]) for i in 1:num_rounds]
append!(miss_class_all,[final_miss_class])
append!(models_all,[fin_model])
print(miss_class_all)
end


#Final Label Prediction:

Hope_fin=hcat(Hope,To_be_conv)
Predicted_Labels=[mode.(MLJ.predict(models_all[i],Hope_fin[missing_index[i],Not(Labels)])) for i in 1:length(Labels)]



#Tuning for a later period
r1 = range(xgb_model, :num_iterations, lower=50, upper=100)
r2 = range(xgb_model, :min_data_in_leaf, lower=2, upper=10)
r3 = range(xgb_model, :learning_rate, lower=1e-1, upper=1e0)
tm = TunedModel(model=xgb_model, tuning=Grid(resolution=5),
                resampling=CV(rng=StableRNG(123)), ranges=[r1,r2,r3],
                measure= rmse)
mach = machine(tm, x,Vector(y_hat))
fin_model=MLJ.fit!(mach,rows=train)




