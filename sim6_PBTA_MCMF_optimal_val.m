%% This script renders a graph that compares PBTA and MCMF optimal values
%Preparing the data
N_min = 80;
N_max = 100;
N_stepSize = 5;

M_min = 50;
M_max = 50;
M_stepSize = 5;

ones_percentage_step_size = 0.2;
ones_percentages = 0.7:0.1:1;

number_of_simulations = 10;
checkConstraints = true;
enable_true_benchmark = false;


 
dataObj = struct();
dataObj.run_MCMF = true;
dataObj.N = N_max;
dataObj.M = M_max;
dataObj.numOfVars = dataObj.N * dataObj.M;
%Communication model parameters
    
dataObj.trans_power = 50 * 1e-3; %50 mWatt;
dataObj.path_loss_exp = 2;
dataObj.sigma_sq = 1e-11;
dataObj.controller_bandwidth = 10e6;

%Reliablity level threshold


dataObj.worker_CPU_FromVal = 2e9;
dataObj.worker_CPU_ToVal = 5e9;
dataObj.workers_freqs = dataObj.worker_CPU_FromVal + (dataObj.worker_CPU_ToVal - dataObj.worker_CPU_FromVal) * rand(1, dataObj.N);  % size  = N
%dataObj.workers_freqs = round(dataObj.workers_freqs, 2);

%Workers maximum number of tasks

dataObj.worker_max_tasks_fromval = 1;
dataObj.worker_max_tasks_toval = 1;
dataObj.workers_max_tasks = dataObj.worker_max_tasks_fromval + (dataObj.worker_max_tasks_toval - dataObj.worker_max_tasks_fromval) * rand(1, dataObj.N);  % size  = N
dataObj.workers_max_tasks = round(dataObj.workers_max_tasks);

%Workers distance from the controller

dataObj.worker_distances_fromval = 5;
dataObj.worker_distances_toval = 50;
dataObj.workers_distances = dataObj.worker_distances_fromval + (dataObj.worker_distances_toval - dataObj.worker_distances_fromval) * rand(1, dataObj.N);  % size  = N
%dataObj.workers_distances = round(dataObj.workers_distances);

%Workers Rayleigh coefficient

dataObj.workers_rayleigh = exprnd(1, [1, dataObj.N]); %mu = 1 -->unit mean

%Workers max energy
dataObj.max_energy = 2;

%Tasks' Processing Density

dataObj.task_pdensity_fromVal = 1e2;
dataObj.task_pdensity_toVal = 5e2;
dataObj.tasks_pdensity = dataObj.task_pdensity_fromVal + (dataObj.task_pdensity_toVal - dataObj.task_pdensity_fromVal) * rand(1, dataObj.M);  % size  = M
%dataObj.tasks_pdensity = round(dataObj.tasks_pdensity, 2);

%Tasks priorities
dataObj.task_priority_fromval = 1;
dataObj.task_priority_toval = 1;
dataObj.tasks_priorities = dataObj.task_priority_fromval + (dataObj.task_priority_toval - dataObj.task_priority_fromval) * rand(1, dataObj.M);  % size  = M
dataObj.tasks_priorities = round(dataObj.tasks_priorities);


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

dataObj.delay_dividend = 10;
%dataObj.tasks_deadlines = round(dataObj.tasks_deadlines, 2);
dataObj.rel_epsilon = 0.8;
dataObj.replicas_per_task = 3;
dataObj.workers_costs_fromval = 0.5;

%%Workers costs
dataObj.workers_costs_toval = 2;
dataObj.workers_costs = dataObj.workers_costs_fromval + (dataObj.workers_costs_toval - dataObj.workers_costs_fromval) * rand(1, dataObj.N);  % size  = N
dataObj.all_tasks_workers_costs = zeros(dataObj.M, dataObj.N);
for i = 1:dataObj.N
    for j = 1:dataObj.M
        dataObj.all_tasks_workers_costs(j, i) = dataObj.workers_costs(i) .* dataObj.tasks_dataSize(j) .* dataObj.tasks_pdensity(j);
    end
end


%Workers hazard rates
workers_hazards_percentages = [0.8 0.2];
workers_hazards_percentages_corsp_values = [0.9 0.01];
workers_hazards_actual_count = workers_hazards_percentages .* dataObj.N;
dataObj.workers_hazard_rates = zeros(1, dataObj.N);
strt_idx = 1;
for i = 1:length(workers_hazards_actual_count)
    dataObj.workers_hazard_rates(1, strt_idx:workers_hazards_actual_count(i)) = workers_hazards_percentages_corsp_values(i);
    if (i + 1 <= length(workers_hazards_actual_count))
        workers_hazards_actual_count(i + 1) = workers_hazards_actual_count(i) + workers_hazards_actual_count(i + 1);
    end
    strt_idx = workers_hazards_actual_count(i) + 1;
end



% dataObj.worker_hazzard_rate_fromval = 0.001;
% dataObj.worker_hazzard_rate_toval = 0.05;
% dataObj.workers_hazard_rates = dataObj.worker_hazzard_rate_fromval + (dataObj.worker_hazzard_rate_toval - dataObj.worker_hazzard_rate_fromval) * rand(1, dataObj.N);  % size  = N
%dataObj.workers_hazard_rates = round(dataObj.workers_hazard_rates);

all_results = cell(1, length(ones_percentages));

%Workers history
dataObj.workers_history_length = 100;
dataObj.workers_history_percentageOfOnes_fromval = 0;
dataObj.workers_history_percentageOfOnes_toval = 1;
dataObj.workers_history_percentageOfOnes = dataObj.workers_history_percentageOfOnes_fromval + (dataObj.workers_history_percentageOfOnes_toval - dataObj.workers_history_percentageOfOnes_fromval) * rand(1, dataObj.N);  % size  = N
dataObj.workers_history_numberOfOnes_per_worker = round(dataObj.workers_history_length * dataObj.workers_history_percentageOfOnes);
dataObj.workers_history = cell(1, dataObj.N);
for i=1:dataObj.N
    % Make initial signal with proper number of 0's and 1's.
    dataObj.workers_history{i} = [ones(1, dataObj.workers_history_numberOfOnes_per_worker(i)), zeros(1, dataObj.workers_history_length - dataObj.workers_history_numberOfOnes_per_worker(i))];
    % Scramble them up with randperm
    dataObj.workers_history{i} = dataObj.workers_history{i}(randperm(length(dataObj.workers_history{i})));
    % Count them just to prove it
    numOnes = sum(dataObj.workers_history{i});
end

%Beta distribution per worker
dataObj.workers_history_alphas = zeros(1, dataObj.N);
dataObj.workers_history_betas = zeros(1, dataObj.N);
dataObj.workers_good_rep_cdf = zeros(1, dataObj.N);
dataObj.workers_bad_rep_cdf = zeros(1, dataObj.N);

for i = 1:dataObj.N
    dataObj.workers_history_alphas(i) = sum(dataObj.workers_history{i} > 0);
    dataObj.workers_history_betas(i) = sum(dataObj.workers_history{i} == 0);
end
%Beta distribution expected value per worker
dataObj.workers_history_expected_vals = zeros(1, dataObj.N);
for i = 1:dataObj.N
    dataObj.workers_history_expected_vals(i) = dataObj.workers_history_alphas(i) ./ (dataObj.workers_history_alphas(i) + dataObj.workers_history_betas(i));
    dataObj.workers_good_rep_cdf(i) = betacdf(0.7, dataObj.workers_history_alphas(i), dataObj.workers_history_betas(i), "upper");
    dataObj.workers_bad_rep_cdf(i) = betacdf(0.7, dataObj.workers_history_alphas(i), dataObj.workers_history_betas(i));
end



%% PBTA result
sim = struct();
dataObj.worker_fitness_fn = @(rel_battery, worker_reputation, worker_cost, ... 
        task_data_size, task_processing_density) (exp(rel_battery .^2) + exp(worker_reputation .^ 2)) ./ (worker_cost);
dataObj.worker_fitness_fn_matlab = @(rel_battery, worker_reputation, worker_cost, ... 
    task_data_size, task_processing_density) 1 .* (exp(rel_battery .^2) + exp(worker_reputation .^ 2))  ./ (worker_cost);
sim.PBTA_result = PBTA_simulation(N_min, N_max, N_stepSize, M_min, ... 
    M_max, M_stepSize, number_of_simulations, dataObj, checkConstraints);



% 
% %RepMax
% all_results{strt_idx}.repMax_result = RepMax_simulation(N_min, N_max, N_stepSize, M_min, ...
% M_max, M_stepSize, number_of_simulations, dataObj, checkConstraints);
% 
% %TRUE
% if (enable_true_benchmark)
%     all_results{strt_idx}.true_result = TREED_battery_aware_simulation(N_min, N_max, N_stepSize, M_min, M_max,...
%         M_stepSize, number_of_simulations, dataObj, checkConstraints);
% end




N = ceil((N_max - N_min + 1) ./ N_stepSize);
M = ceil((M_max - M_min + 1) ./ M_stepSize);
n_vector = N_min:N_stepSize:N_max;
m_vector = M_min:M_stepSize:M_max;
failure_percentage = 0.5;


PBTA_percentages_average = zeros(N, M);
PBTA_MCMF_percentages_average = zeros(N, M);


for sim_count=1:number_of_simulations
    for n=1:N
        for m=1:M
            try 
                PBTA_percentages_average(n, m) = double(0);
                PBTA_MCMF_percentages_average(n, m) = double(0);
                current_m = sim.PBTA_result{n,m}.dataObj.M;
                current_n = sim.PBTA_result{n,m}.dataObj.N;
                
                sim.Ms(n,m) = current_m;
                sim.Ns(n, m) = current_n;
                current_dataObj = sim.PBTA_result{n, m}.dataObj;


                %% Step 1: prepare decision vars
                
                
                %PBTA
                PBTA_x = sim.PBTA_result{n, m}.all_sims{sim_count}.optimal_solution.x;
                PBTA_x_reshaped = reshape(PBTA_x, [current_m, current_n]);
                PBTA_replicas = sum(PBTA_x_reshaped, 2);


                %PBTA MCMF
                PBTA_MCMF_x = double(sim.PBTA_result{n, m}.all_sims{sim_count}.MCMF_result.X)';
                PBTA_MCMF_x_reshaped = reshape(PBTA_MCMF_x, [current_m, current_n]);
                PBTA_MCMF_replicas = sum(PBTA_MCMF_x_reshaped, 2);

                
                %debugging
                pbta_optimal_val = current_dataObj.objectiveFunction_matlab * PBTA_x;
                mcmf_optimal_val = current_dataObj.objectiveFunction_matlab * PBTA_MCMF_x;
                pbta_total_replicas = sum(PBTA_replicas);
                mcmf_total_replicas = sum(PBTA_MCMF_replicas);
                
                PBTA_percentages_average(n, m) = PBTA_percentages_average(n, m) + pbta_optimal_val;
                PBTA_MCMF_percentages_average(n, m) = PBTA_MCMF_percentages_average(n, m) + mcmf_optimal_val;

            catch ex
                disp('error occured!');
                rethrow(ex)
            end
        end%end tasks for loop
    end%simulations for loop
end%rep percentage for loop
 
%calculate the averages
PBTA_percentages_average = PBTA_percentages_average ./ number_of_simulations;
PBTA_MCMF_percentages_average = PBTA_MCMF_percentages_average ./ number_of_simulations;

%sort columns
sort_averages = true;
if (sort_averages)
    PBTA_percentages_average = sort(PBTA_percentages_average, 2);
    PBTA_MCMF_percentages_average = sort(PBTA_MCMF_percentages_average, 2);
end


PBTA_plot = '';
PBTA_MCMF_plot = '';

if (N > M)
    for n=1:N
        current_n = n_vector(n);
        PBTA_plot = strcat(PBTA_plot, '(', num2str(current_n),', ', num2str(PBTA_percentages_average(n, 1)), ')');
        PBTA_MCMF_plot = strcat(PBTA_MCMF_plot, '(', num2str(current_n),', ', num2str(PBTA_MCMF_percentages_average(n, 1)), ')');
    end
    disp(strcat('###Optimal values: PBTA VS MCMF - Varying Number of Workers', ' ###'));
    disp('PBTA Plot: ');
    disp(PBTA_plot);
    disp('-------------------------------------------------------');
    disp('PBTA MCMF Plot: ');
    disp(PBTA_MCMF_plot);
    disp('-------------------------------------------------------');
else 
    for m=1:M
        current_m = m_vector(m);
        PBTA_plot = strcat(PBTA_plot, '(', num2str(current_m),', ', num2str(PBTA_percentages_average(m, 1)), ')');
        PBTA_MCMF_plot = strcat(PBTA_MCMF_plot, '(', num2str(current_m),', ', num2str(PBTA_MCMF_percentages_average(m, 1)), ')');
    end
    disp(strcat('###Optimal values: PBTA VS MCMF - Varying Number of Tasks', ' ###'));
    disp('PBTA Plot: ');
    disp(PBTA_plot);
    disp('-------------------------------------------------------');
    disp('PBTA MCMF Plot: ');
    disp(PBTA_MCMF_plot);
    disp('-------------------------------------------------------');
end