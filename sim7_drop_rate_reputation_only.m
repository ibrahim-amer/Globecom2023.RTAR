%% This script renders a graph that compares PBTA, MCMF, RepMax, TRUE, PBTA-REPD and MCMF-REPD while varying the reputation of the workers and the number of tasks.
%Preparing the data
N_min = 4;
N_max = 4;
N_stepSize = 5;

M_min = 2;
M_max = 2;
M_stepSize = 10;

ones_percentage_step_size = 0.2;
ones_percentages = 0.8;

number_of_simulations = 10;
checkConstraints = true;
enable_true_benchmark = false;

 
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

dataObj.worker_hazzard_rate_fromval = 0.05;
dataObj.worker_hazzard_rate_toval = 0.8;
dataObj.workers_hazard_rates = dataObj.worker_hazzard_rate_fromval + (dataObj.worker_hazzard_rate_toval - dataObj.worker_hazzard_rate_fromval) * rand(1, dataObj.N);  % size  = N
%dataObj.workers_hazard_rates = round(dataObj.workers_hazard_rates);

all_results = cell(1, length(ones_percentages));
ctr = 1;
for ones_percentage=1:length(ones_percentages)
    %Workers history
    dataObj.workers_history_length = 1000;
    dataObj.workers_history_percentageOfOnes_fromval = 0;
    dataObj.workers_history_percentageOfOnes_toval = ones_percentages(ones_percentage);
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



   
    all_results{ctr} = struct();
    %% PBTA and MCMF: reliability and reputation are considered
    dataObj.worker_fitness_fn = @(rel_battery, worker_reputation, worker_cost, ... 
            task_data_size, task_processing_density) (exp(rel_battery .^2) + exp(worker_reputation .^ 2)) ./ (worker_cost);
    dataObj.worker_fitness_fn_matlab = @(rel_battery, worker_reputation, worker_cost, ... 
        task_data_size, task_processing_density) 1 .* (exp(rel_battery .^2) + exp(worker_reputation .^ 2))  ./ (worker_cost);
    all_results{ctr}.PBTA_result = PBTA_simulation(N_min, N_max, N_stepSize, M_min, ... 
        M_max, M_stepSize, number_of_simulations, dataObj, checkConstraints);
    
    %% PBTA and MCMF: worker's reputation is only considered
    dataObj.worker_fitness_fn = @(rel_battery, worker_reputation, worker_cost, ... 
            task_data_size, task_processing_density) (exp(worker_reputation .^ 2)) ./ (worker_cost);
    dataObj.worker_fitness_fn_matlab = @(rel_battery, worker_reputation, worker_cost, ... 
        task_data_size, task_processing_density) 1 .* (exp(worker_reputation .^ 2))  ./ (worker_cost);
    all_results{ctr}.PBTA_result_rep_only = PBTA_simulation(N_min, N_max, N_stepSize, M_min, ... 
        M_max, M_stepSize, number_of_simulations, dataObj, checkConstraints);
    

    %% Change objective function by dropping reliability and reputation constraints.
    dataObj.worker_fitness_fn = @(rel_battery, worker_reputation, worker_cost, ... 
            task_data_size, task_processing_density) 1 ./ (worker_cost);
    dataObj.worker_fitness_fn_matlab = @(rel_battery, worker_reputation, worker_cost, ... 
        task_data_size, task_processing_density) 1 ./ (worker_cost);

    all_results{ctr}.PBTA_cost_only = PBTA_simulation(N_min, N_max, N_stepSize, M_min, ... 
        M_max, M_stepSize, number_of_simulations, dataObj, checkConstraints);
    
    %RepMax
    all_results{ctr}.repMax_result = RepMax_simulation(N_min, N_max, N_stepSize, M_min, ...
    M_max, M_stepSize, number_of_simulations, dataObj, checkConstraints);

    %TRUE
    if (enable_true_benchmark)
        all_results{ctr}.true_result = TREED_battery_aware_simulation(N_min, N_max, N_stepSize, M_min, M_max,...
            M_stepSize, number_of_simulations, dataObj, checkConstraints);
    end

    
    ctr = ctr + 1;

end

N = ceil((N_max - N_min + 1) ./ N_stepSize);
M = ceil((M_max - M_min + 1) ./ M_stepSize);
n_vector = N_min:N_stepSize:N_max;
m_vector = M_min:M_stepSize:M_max;
failure_percentage = 0.5;


rep_max_average_task_drop_rate = zeros(N, M, length(all_results));
true_average_task_drop_rate = zeros(N, M, length(all_results));

PBTA_average_task_drop_rate = zeros(N, M, length(all_results));
PBTA_MCMF_average_task_drop_rate = zeros(N, M, length(all_results));

PBTA_rep_only_average_task_drop_rate = zeros(N, M, length(all_results));
PBTA_MCMF_rep_only_average_task_drop_rate = zeros(N, M, length(all_results));

PBTA_cost_only_average_task_drop_rate = zeros(N, M, length(all_results));
PBTA_MCMF_cost_only_average_task_drop_rate = zeros(N, M, length(all_results));

for i=1:length(all_results)
    for sim_count=1:number_of_simulations
        for n = 1:N
            for m=1:M
                try 
                    current_m = all_results{i}.PBTA_result{n,m}.dataObj.M;
                    current_n = all_results{i}.PBTA_result{n,m}.dataObj.N;
                    current_dataObj = all_results{i}.PBTA_result{n, m}.dataObj;
                    current_dataObj_rep_only = all_results{i}.PBTA_result_rep_only{n, m}.dataObj;
                    current_dataObj_cost_only = all_results{i}.PBTA_cost_only{n, m}.dataObj;


                    %% Step 1: prepare decision vars
                    %% RepMax
                    rep_max_x = all_results{i}.repMax_result{n,m}.repmax.all_sims{sim_count}.x;
                    rep_max_x = rep_max_x(1:(length(rep_max_x) - current_m));

                    rep_max_x_reshaped = reshape(rep_max_x, [current_m, current_n]);
                    rep_max_replicas = sum(rep_max_x_reshaped, 2);


                    %% True
                    if (enable_true_benchmark)
                        true_x = all_results{i}.true_result{1, m}.all_sims{sim_count}.x;
                        true_x_reshaped = reshape(true_x, [current_m, current_n]);
                        true_replicas = sum(true_x_reshaped, 2);
                    end

                    %% PBTA
                    %PBTA
                    PBTA_x = all_results{i}.PBTA_result{1, m}.all_sims{sim_count}.optimal_solution.x;
                    PBTA_x_reshaped = reshape(PBTA_x, [current_m, current_n]);
                    PBTA_replicas = sum(PBTA_x_reshaped, 2);


                    %PBTA MCMF
                    PBTA_MCMF_x = double(all_results{i}.PBTA_result{n, m}.all_sims{sim_count}.MCMF_result.X);
                    PBTA_MCMF_x_reshaped = reshape(PBTA_MCMF_x, [current_m, current_n]);
                    PBTA_MCMF_replicas = sum(PBTA_MCMF_x_reshaped, 2);
                    
                    

                    %% PBTA reputation only
                    %PBTA rep only
                    PBTA_rep_only_x = all_results{i}.PBTA_result_rep_only{n, m}.all_sims{sim_count}.optimal_solution.x;
                    PBTA_rep_only_x_reshaped = reshape(PBTA_rep_only_x, [current_m, current_n]);
                    PBTA_rep_only_replicas = sum(PBTA_rep_only_x_reshaped, 2);
                    
                    %PBTA MCMF rep only
                    PBTA_MCMF_rep_only_x = double(all_results{i}.PBTA_result_rep_only{n, m}.all_sims{sim_count}.MCMF_result.X);
                    PBTA_MCMF_rep_only_x_reshaped = reshape(PBTA_MCMF_rep_only_x, [current_m, current_n]);
                    PBTA_MCMF_rep_only_replicas = sum(PBTA_MCMF_rep_only_x_reshaped, 2);        
                    
                    
                    %% PBTA Cost only
                    %PBTA cost only
                    PBTA_cost_only_x = all_results{i}.PBTA_cost_only{n, m}.all_sims{sim_count}.optimal_solution.x;
                    PBTA_cost_only_x_reshaped = reshape(PBTA_cost_only_x, [current_m, current_n]);
                    PBTA_cost_only_replicas = sum(PBTA_cost_only_x_reshaped, 2);
                    
                    %PBTA MCMF cost only
                    PBTA_MCMF_cost_only_x = double(all_results{i}.PBTA_cost_only{n, m}.all_sims{sim_count}.MCMF_result.X);
                    PBTA_MCMF_cost_only_x_reshaped = reshape(PBTA_MCMF_cost_only_x, [current_m, current_n]);
                    PBTA_MCMF_cost_only_replicas = sum(PBTA_MCMF_cost_only_x_reshaped, 2);           
                    


                    %% Step2: Calculate failure probability
                    failure_rel_probs = 1 - current_dataObj.workers_tasks_rel_prop;
                    failure_rel_probs = reshape(failure_rel_probs, [current_m, current_n]);
                    %workers_bad_rep_cdf = reshape(current_dataObj.workers_bad_rep_cdf, [1, current_n]);
                    workers_bad_rep_cdf = reshape(current_dataObj.workers_history_expected_vals, [1, current_n]);
                    failure_rep_probs = ones(current_m, current_n) .* (1 - workers_bad_rep_cdf);
                    joint_failure_probs = failure_rel_probs .* failure_rep_probs;

                    %% RepMax
                    rep_max_tasks_fail_prob = (rep_max_x_reshaped .* joint_failure_probs) > failure_percentage;
                    rep_max_nodes_failed = sum(rep_max_tasks_fail_prob, 2);
                    rep_max_actual_nodes = abs(rep_max_nodes_failed - rep_max_replicas);
                    rep_max_status = rep_max_actual_nodes == 0;
                    rep_max_percentage = sum(rep_max_status) ./ current_m;

                    rep_max_average_task_drop_rate(n, m, i) = rep_max_average_task_drop_rate(n, m, i) + rep_max_percentage;


                    %% TRUE
                    if (enable_true_benchmark)
                        TRUE_tasks_fail_prob = (true_x_reshaped .* joint_failure_probs) > failure_percentage;
                        TRUE_nodes_failed = sum(TRUE_tasks_fail_prob, 2);
                        TRUE_actual_nodes = abs(TRUE_nodes_failed - true_replicas);
                        TRUE_status = TRUE_actual_nodes == 0;
                        TRUE_percentage = sum(TRUE_status) ./ current_m;
                    end

                    if (enable_true_benchmark)
                        true_average_task_drop_rate(n, m, i) = true_average_task_drop_rate(n, m, i) + TRUE_percentage;
                    end

                    %% PBTA
                    %%PBTA
                    PBTA_tasks_fail_prob = (PBTA_x_reshaped .* joint_failure_probs) > failure_percentage;
                    PBTA_nodes_failed = sum(PBTA_tasks_fail_prob, 2);
                    PBTA_actual_nodes = abs(PBTA_nodes_failed - PBTA_replicas);
                    PBTA_status = PBTA_actual_nodes == 0;
                    PBTA_percentage = sum(PBTA_status) ./ current_m;


                    PBTA_average_task_drop_rate(n, m, i) = PBTA_average_task_drop_rate(n, m, i) + PBTA_percentage;

                    %%PBTA MCMF
                    PBTA_MCMF_tasks_fail_prob = (PBTA_MCMF_x_reshaped .* joint_failure_probs) > failure_percentage;
                    PBTA_MCMF_nodes_failed = sum(PBTA_MCMF_tasks_fail_prob, 2);
                    PBTA_MCMF_actual_nodes = abs(PBTA_MCMF_nodes_failed - PBTA_MCMF_replicas);
                    PBTA_MCMF_status = PBTA_MCMF_actual_nodes == 0;
                    PBTA_MCMF_percentage = sum(PBTA_MCMF_status) ./ current_m;


                    PBTA_MCMF_average_task_drop_rate(n, m, i) = PBTA_MCMF_average_task_drop_rate(n, m, i) + PBTA_MCMF_percentage;

                    %% PBTA reputation only
                    %%PBTA rep only
                    PBTA_rep_only_tasks_fail_prob = (PBTA_rep_only_x_reshaped .* joint_failure_probs) > failure_percentage;
                    PBTA_rep_only_nodes_failed = sum(PBTA_rep_only_tasks_fail_prob, 2);
                    PBTA_rep_only_actual_nodes = abs(PBTA_rep_only_nodes_failed - PBTA_rep_only_replicas);
                    PBTA_rep_only_status = PBTA_rep_only_actual_nodes == 0;
                    PBTA_rep_only_percentage = sum(PBTA_rep_only_status) ./ current_m;


                    PBTA_rep_only_average_task_drop_rate(n, m, i) = PBTA_rep_only_average_task_drop_rate(n, m, i) + PBTA_rep_only_percentage;

                    %%PBTA MCMF rep only
                    PBTA_MCMF_rep_only_tasks_fail_prob = (PBTA_MCMF_rep_only_x_reshaped .* joint_failure_probs) > failure_percentage;
                    PBTA_MCMF_rep_only_nodes_failed = sum(PBTA_MCMF_rep_only_tasks_fail_prob, 2);
                    PBTA_MCMF_rep_only_actual_nodes = abs(PBTA_MCMF_rep_only_nodes_failed - PBTA_MCMF_rep_only_replicas);
                    PBTA_MCMF_rep_only_status = PBTA_MCMF_rep_only_actual_nodes == 0;
                    PBTA_MCMF_rep_only_percentage = sum(PBTA_MCMF_rep_only_status) ./ current_m;


                    PBTA_MCMF_rep_only_average_task_drop_rate(n, m, i) = PBTA_MCMF_rep_only_average_task_drop_rate(n, m, i) + PBTA_MCMF_rep_only_percentage;
                    
                    %% PBTA Cost only
                    %%PBTA cost only
                    PBTA_cost_only_tasks_fail_prob = (PBTA_rep_only_x_reshaped .* joint_failure_probs) > failure_percentage;
                    PBTA_cost_only_nodes_failed = sum(PBTA_cost_only_tasks_fail_prob, 2);
                    PBTA_cost_only_actual_nodes = abs(PBTA_cost_only_nodes_failed - PBTA_rep_only_replicas);
                    PBTA_cost_only_status = PBTA_cost_only_actual_nodes == 0;
                    PBTA_cost_only_percentage = sum(PBTA_cost_only_status) ./ current_m;


                    PBTA_cost_only_average_task_drop_rate(n, m, i) = PBTA_cost_only_average_task_drop_rate(n, m, i) + PBTA_cost_only_percentage;

                    %%PBTA MCMF cost only
                    PBTA_MCMF_cost_only_tasks_fail_prob = (PBTA_MCMF_rep_only_x_reshaped .* joint_failure_probs) > failure_percentage;
                    PBTA_MCMF_cost_only_nodes_failed = sum(PBTA_MCMF_cost_only_tasks_fail_prob, 2);
                    PBTA_MCMF_cost_only_actual_nodes = abs(PBTA_MCMF_cost_only_nodes_failed - PBTA_MCMF_rep_only_replicas);
                    PBTA_MCMF_cost_only_status = PBTA_MCMF_cost_only_actual_nodes == 0;
                    PBTA_MCMF_cost_only_percentage = sum(PBTA_MCMF_cost_only_status) ./ current_m;


                    PBTA_MCMF_cost_only_average_task_drop_rate(n, m, i) = PBTA_MCMF_cost_only_average_task_drop_rate(n, m, i) + PBTA_MCMF_cost_only_percentage;

                catch ex
                    disp('error occured!');
                    rethrow(ex)
                end%try catch block
            end%end tasks for loop
        end%workers for loop
    end%simulations for loop
end%rep percentage for loop
 
%calculate the averages
rep_max_average_task_drop_rate = rep_max_average_task_drop_rate ./ number_of_simulations;
if (enable_true_benchmark)
    true_average_task_drop_rate = true_average_task_drop_rate ./ number_of_simulations;
end
PBTA_average_task_drop_rate = PBTA_average_task_drop_rate ./ number_of_simulations;
PBTA_MCMF_average_task_drop_rate = PBTA_MCMF_average_task_drop_rate ./ number_of_simulations;
PBTA_rep_only_average_task_drop_rate = PBTA_rep_only_average_task_drop_rate ./ number_of_simulations;
PBTA_MCMF_rep_only_average_task_drop_rate = PBTA_MCMF_rep_only_average_task_drop_rate ./ number_of_simulations;
PBTA_cost_only_average_task_drop_rate = PBTA_cost_only_average_task_drop_rate ./ number_of_simulations;
PBTA_MCMF_cost_only_average_task_drop_rate = PBTA_MCMF_cost_only_average_task_drop_rate ./ number_of_simulations;
%sort columns
sort_averages = true;
if (sort_averages)
    
    rep_max_average_task_drop_rate = sort(rep_max_average_task_drop_rate, 2);
    if (enable_true_benchmark)
        true_average_task_drop_rate = sort(true_average_task_drop_rate, 2);
    end
    if (N >= M)
        PBTA_average_task_drop_rate = sort(PBTA_average_task_drop_rate, 1);
        PBTA_MCMF_average_task_drop_rate = sort(PBTA_MCMF_average_task_drop_rate, 1);
        PBTA_rep_only_average_task_drop_rate = sort(PBTA_rep_only_average_task_drop_rate, 1);
        PBTA_MCMF_rep_only_average_task_drop_rate = sort(PBTA_MCMF_rep_only_average_task_drop_rate, 1);
        PBTA_cost_only_average_task_drop_rate = sort(PBTA_cost_only_average_task_drop_rate, 1);
        PBTA_MCMF_cost_only_average_task_drop_rate = sort(PBTA_MCMF_cost_only_average_task_drop_rate, 1);
    elseif (M > N)
        PBTA_average_task_drop_rate = sort(PBTA_average_task_drop_rate, 2);
        PBTA_MCMF_average_task_drop_rate = sort(PBTA_MCMF_average_task_drop_rate, 2);
        PBTA_rep_only_average_task_drop_rate = sort(PBTA_rep_only_average_task_drop_rate, 2);
        PBTA_MCMF_rep_only_average_task_drop_rate = sort(PBTA_MCMF_rep_only_average_task_drop_rate, 2);
        PBTA_cost_only_average_task_drop_rate = sort(PBTA_cost_only_average_task_drop_rate, 2);
        PBTA_MCMF_cost_only_average_task_drop_rate = sort(PBTA_MCMF_cost_only_average_task_drop_rate, 2);
    end    
end

PBTA_plot = '';
PBTA_MCMF_plot = '';

PBTA_rep_only_plot = '';
PBTA_MCMF_rep_only_plot = '';

PBTA_cost_only_plot = '';
PBTA_MCMF_cost_only_plot = '';

repMax_plot = '';
true_plot = '';

for ones_percentage=1:length(ones_percentages)
    repMax_plot = '';
    PBTA_plot = '';
    PBTA_MCMF_plot = '';

    PBTA_rep_only_plot = '';
    PBTA_MCMF_rep_only_plot = '';

    PBTA_cost_only_plot = '';
    PBTA_MCMF_cost_only_plot = '';
    current_one_percent = ones_percentages(ones_percentage);
    if (N >= M)
        for n = 1:N
            current_n = n_vector(n);
            repMax_plot = strcat(repMax_plot, '(', num2str(current_n),', ', num2str(rep_max_average_task_drop_rate(n, 1, ones_percentage)), ')');
            if (enable_true_benchmark)
                true_plot = strcat(true_plot, '(', num2str(current_n),', ', num2str(true_average_task_drop_rate(n, 1, ones_percentage)), ')');
            end
            PBTA_plot = strcat(PBTA_plot, '(', num2str(current_n),', ', num2str(PBTA_average_task_drop_rate(ones_percentage, n, 1)), ')');
            PBTA_MCMF_plot = strcat(PBTA_MCMF_plot, '(', num2str(current_n),', ', num2str(PBTA_MCMF_average_task_drop_rate(n, 1, ones_percentage)), ')');
            
            PBTA_rep_only_plot = strcat(PBTA_rep_only_plot, '(', num2str(current_n),', ', num2str(PBTA_rep_only_average_task_drop_rate(n, 1, ones_percentage)), ')');
            PBTA_MCMF_rep_only_plot = strcat(PBTA_MCMF_rep_only_plot, '(', num2str(current_n),', ', num2str(PBTA_MCMF_rep_only_average_task_drop_rate(ones_percentage, n, 1)), ')');
            
            PBTA_cost_only_plot = strcat(PBTA_cost_only_plot, '(', num2str(current_n),', ', num2str(PBTA_cost_only_average_task_drop_rate(n, 1, ones_percentage)), ')');
            PBTA_MCMF_cost_only_plot = strcat(PBTA_MCMF_cost_only_plot, '(', num2str(current_n),', ', num2str(PBTA_MCMF_cost_only_average_task_drop_rate(n, 1, ones_percentage)), ')');
        end
    else
        for m = 1:M
            current_m = m_vector(n);
            repMax_plot = strcat(repMax_plot, '(', num2str(current_m),', ', num2str(rep_max_average_task_drop_rate(1, m, ones_percentage)), ')');
            if (enable_true_benchmark)
                true_plot = strcat(true_plot, '(', num2str(current_m),', ', num2str(true_average_task_drop_rate(1, m, ones_percentage)), ')');
            end
            PBTA_plot = strcat(PBTA_plot, '(', num2str(current_m),', ', num2str(PBTA_average_task_drop_rate(1, m, ones_percentage)), ')');
            PBTA_MCMF_plot = strcat(PBTA_MCMF_plot, '(', num2str(current_m),', ', num2str(PBTA_MCMF_average_task_drop_rate(1, m, ones_percentage)), ')');
            
            PBTA_rep_only_plot = strcat(PBTA_rep_only_plot, '(', num2str(current_m),', ', num2str(PBTA_rep_only_average_task_drop_rate(1, m, ones_percentage)), ')');
            PBTA_MCMF_rep_only_plot = strcat(PBTA_MCMF_rep_only_plot, '(', num2str(current_m),', ', num2str(PBTA_MCMF_rep_only_average_task_drop_rate(1, m, ones_percentage)), ')');
            
            PBTA_cost_only_plot = strcat(PBTA_cost_only_plot, '(', num2str(current_m),', ', num2str(PBTA_cost_only_average_task_drop_rate(1, m, ones_percentage)), ')');
            PBTA_MCMF_cost_only_plot = strcat(PBTA_MCMF_cost_only_plot, '(', num2str(current_m),', ', num2str(PBTA_MCMF_cost_only_average_task_drop_rate(1, m, ones_percentage)), ')');
        end
    end%end if else
    disp(strcat('###Average task drop rate ones percentage = ', num2str(current_one_percent), ' ###'));
    disp('PBTA Plot: ');
    disp(PBTA_plot);
    disp('-------------------------------------------------------');
    disp('PBTA MCMF Plot: ');
    disp(PBTA_MCMF_plot);
    disp('-------------------------------------------------------');
    disp('PBTA rep only Plot: ');
    disp(PBTA_rep_only_plot);
    disp('-------------------------------------------------------');
    disp('PBTA MCMF rep only Plot: ');
    disp(PBTA_MCMF_rep_only_plot);
    disp('-------------------------------------------------------');
    disp('PBTA cost only Plot: ');
    disp(PBTA_cost_only_plot);
    disp('-------------------------------------------------------');
    disp('PBTA MCMF cost only Plot: ');
    disp(PBTA_MCMF_cost_only_plot);
    disp('-------------------------------------------------------');
    disp('RepMax Plot: ');
    disp(repMax_plot);
    disp('-------------------------------------------------------');
    if (enable_true_benchmark)
        disp('true_plot Plot: ');
        disp(true_plot);
    end
    disp('################################################');
end%end ones_percentages loop


