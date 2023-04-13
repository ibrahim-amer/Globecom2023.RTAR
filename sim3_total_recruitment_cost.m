%% This script renders a graph that compares the total recruitment cost for the methods True, RepMax, PBTA, and MCMF
%Preparing the data
N_min = 50;
N_max = 100;
N_stepSize = 5;

M_min = 20;
M_max = 20;
M_stepSize = 5;

number_of_simulations = 10;
checkConstraints = true;
enable_true_benchmark = true;

 
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

dataObj.worker_hazzard_rate_fromval = 0.05;
dataObj.worker_hazzard_rate_toval = 0.2;
dataObj.workers_hazard_rates = dataObj.worker_hazzard_rate_fromval + (dataObj.worker_hazzard_rate_toval - dataObj.worker_hazzard_rate_fromval) * rand(1, dataObj.N);  % size  = N
%dataObj.workers_hazard_rates = round(dataObj.workers_hazard_rates);

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
%             assert(length(dataObj.workers_history{i}) == numOnes, "numOfOnes in worker's history is incorrect!");
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
    

%Workers max energy
dataObj.max_energy = 1;

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

dataObj.task_deadline_fromVal = 5;%was 5
dataObj.task_deadline_toVal = 10;%was 10
dataObj.tasks_deadlines = dataObj.task_deadline_fromVal + (dataObj.task_deadline_toVal - dataObj.task_deadline_fromVal) * rand(1, dataObj.M);  % size  = M


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


dataObj.worker_fitness_fn = @(rel_battery, worker_reputation, worker_cost, ... 
        task_data_size, task_processing_density) (exp(rel_battery .^2) + exp(worker_reputation .^ 2)) ./ (worker_cost);
dataObj.worker_fitness_fn_matlab = @(rel_battery, worker_reputation, worker_cost, ... 
    task_data_size, task_processing_density) 1 .* (exp(rel_battery .^2) + exp(worker_reputation .^ 2))  ./ (worker_cost);
sim.PBTA_result = PBTA_simulation(N_min, N_max, N_stepSize, M_min, ... 
    M_max, M_stepSize, number_of_simulations, dataObj, checkConstraints);

sim = struct();
sim.PBTA_result = PBTA_simulation(N_min, N_max, N_stepSize, M_min, ... 
    M_max, M_stepSize, number_of_simulations, dataObj, checkConstraints);
if (enable_true_benchmark)
    sim.true_result = TREED_battery_aware_simulation(N_min, N_max, N_stepSize, M_min, M_max,...
        M_stepSize, number_of_simulations, dataObj, checkConstraints);
end
sim.repMax_result = RepMax_simulation(N_min, N_max, N_stepSize, M_min, ...
    M_max, M_stepSize, number_of_simulations, dataObj, checkConstraints);


N = ceil((N_max - N_min + 1) ./ N_stepSize);
M = ceil((M_max - M_min + 1) ./ M_stepSize);
n_vector = N_min:N_stepSize:N_max;
m_vector = M_min:M_stepSize:M_max;
failure_percentage = 0.5;


PBTA_average_total_costs = zeros(N, M);
PBTA_MCMF_average_total_costs = zeros(N, M);
PBTA_priority_dropped_average_total_costs = zeros(N, M);
RepMax_average_total_costs = zeros(N, M);
TRUE_average_total_costs = zeros(N, M);

for sim_count=1:number_of_simulations
    for n=1:N
        for m=1:M       
            PBTA_average_total_costs(n, m) = double(0);
            PBTA_MCMF_average_total_costs(n, m) = double(0);
            PBTA_priority_dropped_average_total_costs(n, m) = double(0);
            RepMax_average_total_costs(n, m) = double(0);
            TRUE_average_total_costs(n, m) = double(0);
            try 
                sim.Ms(n,m) = sim.repMax_result{n,m}.dataObj.M;
                current_m = sim.Ms(n,m);
                sim.Ns(n, m) = sim.repMax_result{n,m}.dataObj.N;
                current_n = sim.Ns(n, m);
                current_dataObj = sim.PBTA_result{n, m}.dataObj;


                %% Step 1: prepare decision vars

                %RepMax
                rep_max_x = sim.repMax_result{n,m}.repmax.all_sims{sim_count}.x;
                rep_max_x = rep_max_x(1:(length(rep_max_x) - current_m));

                rep_sw_x = sim.repMax_result{n,m}.rep_sw.all_sims{sim_count}.x;
                rep_sw_x = rep_sw_x(1:(length(rep_sw_x) - current_m));

                rep_kw_x = sim.repMax_result{n,m}.rep_kw.all_sims{sim_count}.x;
                rep_kw_x = rep_kw_x(1:(length(rep_kw_x) - current_m));


                rep_max_x_reshaped = reshape(rep_max_x, [current_m, current_n]);
                rep_sw_x_reshaped = reshape(rep_sw_x, [current_m, current_n]);
                rep_kw_x_reshaped = reshape(rep_kw_x, [current_m, current_n]);

                rep_max_replicas = sum(rep_max_x_reshaped, 2);
                rep_sw_replicas = sum(rep_sw_x_reshaped, 2);
                rep_kw_replicas = sum(rep_kw_x_reshaped, 2);

                RepMax_tasks_workers_costs = rep_max_x_reshaped .* current_dataObj.all_tasks_workers_costs;
                RepMax_current_total_cost = sum(RepMax_tasks_workers_costs, 'all');

                %True
                if (enable_true_benchmark)
                    true_x = sim.true_result{n, m}.all_sims{sim_count}.x;
                    true_x_reshaped = reshape(true_x, [current_m, current_n]);
                    true_replicas = sum(true_x_reshaped, 2);

                    TRUE_tasks_workers_costs = true_x_reshaped .* current_dataObj.all_tasks_workers_costs;
                    TRUE_current_total_cost = sum(TRUE_tasks_workers_costs, 'all');
                end

                %PBTA
                PBTA_x = sim.PBTA_result{n, m}.all_sims{sim_count}.optimal_solution.x;
                PBTA_x_reshaped = reshape(PBTA_x, [current_m, current_n]);
                PBTA_replicas = sum(PBTA_x_reshaped, 2);

                PBTA_tasks_workers_costs = PBTA_x_reshaped .* current_dataObj.all_tasks_workers_costs;
                PBTA_current_total_cost = sum(PBTA_tasks_workers_costs, 'all');

                %PBTA MCMF
                PBTA_MCMF_x = double(sim.PBTA_result{n, m}.all_sims{sim_count}.MCMF_result.X);
                PBTA_MCMF_x_reshaped = reshape(PBTA_MCMF_x, [current_m, current_n]);
                PBTA_MCMF_replicas = sum(PBTA_MCMF_x_reshaped, 2);

                PBTA_MCMF_tasks_workers_costs = PBTA_MCMF_x_reshaped .* current_dataObj.all_tasks_workers_costs;
                PBTA_MCMF_current_total_cost = sum(PBTA_MCMF_tasks_workers_costs, 'all');

                PBTA_average_total_costs(n, m) = PBTA_average_total_costs(n, m) + PBTA_current_total_cost;
                PBTA_MCMF_average_total_costs(n, m) = PBTA_MCMF_average_total_costs(n, m) + PBTA_MCMF_current_total_cost;
                RepMax_average_total_costs(n, m) = RepMax_average_total_costs(n, m) + RepMax_current_total_cost;
                if (enable_true_benchmark)
                    TRUE_average_total_costs(n, m) = TRUE_average_total_costs(n, m) + TRUE_current_total_cost;
                end

            catch ex
                disp('error occured!');
                rethrow(ex)
            end

         end%Tasks for loop
    end%workers for loop
end%sim for loop


%% Step 3: calculate the averages

%%PBTA
PBTA_average_total_costs = PBTA_average_total_costs ./ number_of_simulations;
%%PBTA MCMF
PBTA_MCMF_average_total_costs = PBTA_MCMF_average_total_costs ./ number_of_simulations;
%%RepMax
RepMax_average_total_costs = RepMax_average_total_costs ./ number_of_simulations;
%%TRUE
if (enable_true_benchmark)
    TRUE_average_total_costs = TRUE_average_total_costs ./ number_of_simulations;
end

average_vectors_size = size(PBTA_average_total_costs);
average_vectors_size(3) = 4; 
all_average_costs = nan(average_vectors_size);
all_average_costs(:, :, 1) = PBTA_average_total_costs;
all_average_costs(:, :, 2) = PBTA_MCMF_average_total_costs;
all_average_costs(:, :, 3) = RepMax_average_total_costs;
all_average_costs(:, :, 4) = TRUE_average_total_costs;

max_average_cost = max(all_average_costs, [], 'all');

disp_div = numel(num2str(ceil(max_average_cost))) - 1;
disp_div = 10 .^ disp_div;
PBTA_average_total_costs = PBTA_average_total_costs ./ disp_div;
PBTA_MCMF_average_total_costs = PBTA_MCMF_average_total_costs ./ disp_div;
RepMax_average_total_costs = RepMax_average_total_costs ./ disp_div;
TRUE_average_total_costs = TRUE_average_total_costs ./ disp_div;

sort_averages = true;
if (sort_averages)
    PBTA_average_total_costs = sort(PBTA_average_total_costs);
    PBTA_MCMF_average_total_costs = sort(PBTA_MCMF_average_total_costs);
    RepMax_average_total_costs = sort(RepMax_average_total_costs);
    TRUE_average_total_costs = sort(TRUE_average_total_costs);
end

RepMax_plot = '';
TRUE_plot = '';
PBTA_plot = '';
PBTA_MCMF_plot = '';
for n=1:N
    for m=1:M
        %% Step 4: build graphs
        current_m = sim.Ms(n,m);
        current_n = sim.Ns(n, m);
        PBTA_plot = strcat(PBTA_plot, '(', num2str(current_n),', ', num2str(PBTA_average_total_costs(n, m)), ')');
        PBTA_MCMF_plot = strcat(PBTA_MCMF_plot, '(', num2str(current_n),', ', num2str(PBTA_MCMF_average_total_costs(n, m)), ')');
        %PBTA_priority_dropped_plot = strcat(PBTA_priority_dropped_plot, '(', num2str(current_m),', ', num2str(PBTA_priority_dropped_average_total_costs(n, m)), ')');
        RepMax_plot = strcat(RepMax_plot, '(', num2str(current_n),', ', num2str(RepMax_average_total_costs(n, m)), ')');
        if (enable_true_benchmark)
            TRUE_plot = strcat(TRUE_plot, '(', num2str(current_n),', ', num2str(TRUE_average_total_costs(n, m)), ')');
        end
    end
end

disp(strcat('###Total recruitment cost, for M = ', num2str(current_m), 'and N = ', num2str(current_n), '###'));
disp('PBTA Plot: ');
disp(PBTA_plot);
disp('PBTA MCMF Plot: ');
disp(PBTA_MCMF_plot);
disp('RepMax Plot: ');
disp(RepMax_plot);
if (enable_true_benchmark)
    disp('TRUE Plot: ');
    disp(TRUE_plot);
end






