%% This script renders a graph for the total allocated replicas over varying the number of workers
disp('#########################################################');
disp('This script renders a graph for the total allocated replicas over varying the number of workers');
N_min = 80;
N_max = 150;
N_stepSize = 10;

M_min = 50;
M_max = 50;
M_stepSize = 1;


number_of_simulations = 1;
checkConstraints = true;
%%init dataObj with the data required for RepMax methedology
dataObj = struct();
dataObj.N = N_max;
dataObj.M = M_max;
dataObj.numOfVars = dataObj.N * dataObj.M;
%Communication model parameters
    
dataObj.trans_power = 50 * 1e-3; %50 mWatt;
dataObj.path_loss_exp = 2;
dataObj.sigma_sq = 1e-11;
dataObj.controller_bandwidth = 10e6;

%Reliablity level threshold


dataObj.worker_CPU_FromVal = 1e9;
dataObj.worker_CPU_ToVal = 4e9;
dataObj.workers_freqs = dataObj.worker_CPU_FromVal + (dataObj.worker_CPU_ToVal - dataObj.worker_CPU_FromVal) * rand(1, dataObj.N);  % size  = N
%dataObj.workers_freqs = round(dataObj.workers_freqs, 2);

%Workers maximum number of tasks

dataObj.worker_max_tasks_fromval = 1;
dataObj.worker_max_tasks_toval = 3;
dataObj.workers_max_tasks = dataObj.worker_max_tasks_fromval + (dataObj.worker_max_tasks_toval - dataObj.worker_max_tasks_fromval) * rand(1, dataObj.N);  % size  = N
dataObj.workers_max_tasks = round(dataObj.workers_max_tasks);

%Workers distance from the controller

dataObj.worker_distances_fromval = 5;
dataObj.worker_distances_toval = 50;
dataObj.workers_distances = dataObj.worker_distances_fromval + (dataObj.worker_distances_toval - dataObj.worker_distances_fromval) * rand(1, dataObj.N);  % size  = N
%dataObj.workers_distances = round(dataObj.workers_distances);

%Workers Rayleigh coefficient

dataObj.workers_rayleigh = exprnd(1, [1, dataObj.N]); %mu = 1 -->unit mean

%Workers hazard rates

dataObj.worker_hazzard_rate_fromval = 0.01;
dataObj.worker_hazzard_rate_toval = 0.5;
dataObj.workers_hazard_rates = dataObj.worker_hazzard_rate_fromval + (dataObj.worker_hazzard_rate_toval - dataObj.worker_hazzard_rate_fromval) * rand(1, dataObj.N);  % size  = N
%dataObj.workers_hazard_rates = round(dataObj.workers_hazard_rates);

%Workers max energy
dataObj.max_energy = 1;

%Tasks' Processing Density

dataObj.task_pdensity_fromVal = 1e2;
dataObj.task_pdensity_toVal = 5e2;
dataObj.tasks_pdensity = dataObj.task_pdensity_fromVal + (dataObj.task_pdensity_toVal - dataObj.task_pdensity_fromVal) * rand(1, dataObj.M);  % size  = M
%dataObj.tasks_pdensity = round(dataObj.tasks_pdensity, 2);


%Tasks data size

dataObj.task_dataSize_fromVal = 1e6;
dataObj.task_dataSize_toVal = 20e6;
dataObj.tasks_dataSize = dataObj.task_dataSize_fromVal + (dataObj.task_dataSize_toVal - dataObj.task_dataSize_fromVal) * rand(1, dataObj.M);  % size  = M
%dataObj.tasks_dataSize = round(dataObj.tasks_dataSize, 2);


%Tasks CPU requirement

dataObj.task_CPU_fromVal = 1e9;
dataObj.task_CPU_toVal = 2e9;
dataObj.tasks_CPU_req = dataObj.task_CPU_fromVal + (dataObj.task_CPU_toVal - dataObj.task_CPU_fromVal) * rand(1, dataObj.M);  % size  = M
%dataObj.tasks_CPU_req = round(dataObj.tasks_dataSize, 2);


%Tasks deadlines - uniformly distributed

dataObj.task_deadline_fromVal = 8;%was 5
dataObj.task_deadline_toVal = 10;%was 10
dataObj.tasks_deadlines = dataObj.task_deadline_fromVal + (dataObj.task_deadline_toVal - dataObj.task_deadline_fromVal) * rand(1, dataObj.M);  % size  = M

dataObj.delay_dividend = 2;
%dataObj.tasks_deadlines = round(dataObj.tasks_deadlines, 2);
dataObj.rel_epsilon = 0.2;
%%
guid = string(java.util.UUID.randomUUID.toString);
signature = strcat('[simulation_results1][N_min =  ', int2str(N_min), ', ', ...
    ' N_max =  ', int2str(N_max), ', ', ...
    ' N_stepSize =  ', int2str(N_stepSize), ', ', ...
    ' M_min =  ', int2str(M_min), ', ', ...
    ' M_max =  ', int2str(M_max),  ', ', ...
    ' M_stepSize =  ', int2str(M_stepSize),  ', ', ...
    ' n_sims =  ', int2str(number_of_simulations),  ']');
guid = strcat(signature, '_', guid);
save_to_file = true;


true_result = TREED_battery_aware_simulation(N_min, N_max, N_stepSize, M_min, M_max,...
    M_stepSize, number_of_simulations, dataObj, checkConstraints);
PBTA_result = PBTA_simulation(N_min, N_max, N_stepSize, M_min, ... 
    M_max, M_stepSize, number_of_simulations, dataObj, checkConstraints);
repMax_result = RepMax_simulation(N_min, N_max, N_stepSize, M_min, M_max,...
    M_stepSize, number_of_simulations, dataObj, checkConstraints);



N = ceil((N_max - N_min + 1) ./ N_stepSize);
M = ceil((M_max - M_min + 1) ./ M_stepSize);
n_vector = N_min:N_stepSize:N_max;
m_vector = M_min:M_stepSize:M_max;
y_axis_opt = zeros(1, N);
y_axis_mcmf = zeros(1, N);
y_axis_repMax = zeros(1, N);
y_axis_true = zeros(1, N);
x_axis = zeros(1, N);
sim1 = struct();
for n=1:N
    for sim_count=1:number_of_simulations
        current_N = PBTA_result{n, 1}.dataObj.N;
        y_axis_opt(n) = y_axis_opt(n) + PBTA_result{n, 1}.all_sims{sim}.optimal_solution.total_number_of_allocated_workers;
        y_axis_mcmf(n) = y_axis_mcmf(n) + PBTA_result{n, 1}.all_sims{sim}.MCMF_result.totalNumberOfAllocatedWorkers;
        y_axis_repMax(n) = y_axis_repMax(n) + repMax_result{n,1}.repmax.stats.averageOptimalVal;

        x = true_result{n,1}.all_sims{1}.x;
        temp_x = reshape(x, [M_min, current_N]);
        num_of_replicas_per_task = sum(temp_x, 2);
        y_axis_true(n) = y_axis_true(n) + sum(num_of_replicas_per_task);

        x_axis(n) = current_N;
    end
end
%Calculate the average
y_axis_opt = y_axis_opt ./ number_of_simulations;
y_axis_mcmf = y_axis_opt ./ number_of_simulations;
y_axis_repMax = y_axis_opt ./ number_of_simulations;
y_axis_true = y_axis_true ./ number_of_simulations;

sort_averages = true;
if (sort_averages)
    y_axis_opt = sort(y_axis_opt);
    y_axis_mcmf = sort(y_axis_mcmf);
    y_axis_repMax = sort(y_axis_repMax);
    y_axis_true = sort(y_axis_true);
end

disp('Total Number of Workers - Fixing number of tasks and varying number of workers');

strresult_opt = '';
strresult_mcmf = '';
strresult_repMax = '';
strresult_true = '';
for i = 1:length(x_axis)
    strresult_opt = strcat(strresult_opt, '(', num2str(x_axis(i)),', ', num2str(y_axis_opt(i)), ')');
    strresult_mcmf = strcat(strresult_mcmf, '(', num2str(x_axis(i)),', ', num2str(y_axis_mcmf(i)), ')');
    strresult_repMax = strcat(strresult_repMax, '(', num2str(x_axis(i)),', ', num2str(y_axis_repMax(i)), ')');
    strresult_true = strcat(strresult_true, '(', num2str(x_axis(i)),', ', num2str(y_axis_true(i)), ')');
end
disp('x_axis: number of workers, y_axis: optimal solution');
disp(strresult_opt);
disp('x_axis: number of workers, y_axis: MCMF');
disp(strresult_mcmf);
disp('x_axis: number of workers, y_axis: RepMax');
disp(strresult_repMax);
disp('x_axis: number of workers, y_axis: True');
disp(strresult_true);

disp('#########################################################');
