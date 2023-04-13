%% This simulation compares PBTA/MCMF and PBTA/MCMF where tasks' priorities are ignored. Number of workers is fixed and the number of tasks is varying
%Preparing the data
N_min = 100;
N_max = 100;
N_stepSize = 10;

M_min = 5;
M_max = 50;
M_stepSize = 5;

number_of_simulations = 1;
checkConstraints = true;

 
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

dataObj.worker_hazzard_rate_fromval = 0.05;
dataObj.worker_hazzard_rate_toval = 1;
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
    dataObj.workers_good_rep_cdf(i) = betacdf(0.7, dataObj.workers_history_alphas(i), dataObj.workers_history_betas(i));
    dataObj.workers_bad_rep_cdf(i) = betacdf(0.3, dataObj.workers_history_alphas(i), dataObj.workers_history_betas(i), "upper");
end



    

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

dataObj.task_deadline_fromVal = 3;%was 5
dataObj.task_deadline_toVal = 5;%was 10
dataObj.tasks_deadlines = dataObj.task_deadline_fromVal + (dataObj.task_deadline_toVal - dataObj.task_deadline_fromVal) * rand(1, dataObj.M);  % size  = M

dataObj.delay_dividend = 1;
%dataObj.tasks_deadlines = round(dataObj.tasks_deadlines, 2);
dataObj.rel_epsilon = 0.8;
dataObj.replicas_per_task = 3;

sim = struct();

sim.PBTA_result = PBTA_simulation(N_min, N_max, N_stepSize, M_min, ... 
    M_max, M_stepSize, number_of_simulations, dataObj, checkConstraints);
true_result = TREED_battery_aware_simulation(N_min, N_max, N_stepSize, M_min, M_max,...
    M_stepSize, number_of_simulations, dataObj, checkConstraints);
repMax_result = RepMax_simulation(N_min, N_max, N_stepSize, M_min, M_max,...
    M_stepSize, number_of_simulations, dataObj, checkConstraints);

%% Change objective function by dropping reliability and reputation constraints.
dataObj.worker_fitness_fn = @(rel_battery, worker_reputation, worker_cost, ... 
task_data_size, task_processing_density) (rel_battery .* worker_reputation) ./ (worker_cost .* task_data_size .* task_processing_density);
ctr = 1;
dataObj.workers_fitness_fns = zeros(1, dataObj.numOfVars);
for i=1:dataObj.N
    for j=1:dataObj.M
        dataObj.workers_fitness_fns(ctr) = dataObj.worker_fitness_fn(dataObj.workers_tasks_rel_prop(ctr), dataObj.workers_history_expected_vals(i), ...
            dataObj.workers_costs(i), dataObj.tasks_dataSize(j), dataObj.tasks_pdensity(j));
        ctr = ctr + 1;
    end
end
dataObj.workers_fitness_fns_inverse = 1 ./ dataObj.workers_fitness_fns;
dataObj.workers_fitness_fns_reshaped = reshape(dataObj.workers_fitness_fns, [dataObj.N, dataObj.M])';
dataObj.obj_fn_elements = zeros(1, dataObj.numOfVars);
ctr = 1;
for i = 1:dataObj.N
    for j = 1:dataObj.M
        dataObj.obj_fn_elements(ctr) = dataObj.workers_fitness_fns(ctr) .* 1;
        ctr = ctr + 1;
    end
end
dataObj.objectiveFunction = dataObj.obj_fn_elements;
dataObj.run_MCMF = false;
sim.PBTA_priority_dropped = PBTA_simulation(N_min, N_max, N_stepSize, M_min, ... 
    M_max, M_stepSize, number_of_simulations, dataObj, checkConstraints);


N = ceil((N_max - N_min + 1) ./ N_stepSize);
M = ceil((M_max - M_min + 1) ./ M_stepSize);
n_vector = N_min:N_stepSize:N_max;
m_vector = M_min:M_stepSize:M_max;
failure_percentage = 0.5;
PBTA_plot = '';
PBTA_MCMF_plot = '';
PBTA_priority_dropped_plot = '';
RepMax_plot = '';
TRUE_plot = '';




PBTA_average_total_costs = zeros(N, M);
PBTA_MCMF_average_total_costs = zeros(N, M);
PBTA_priority_dropped_average_total_costs = zeros(N, M);
RepMax_average_total_costs = zeros(N, M);
TRUE_average_total_costs = zeros(N, M);



for n=1:N
    for m=1:M       
        PBTA_average_total_costs(n, m) = double(0);
        PBTA_MCMF_average_total_costs(n, m) = double(0);
        PBTA_priority_dropped_average_total_costs(n, m) = double(0);
        RepMax_average_total_costs(n, m) = double(0);
        TRUE_average_total_costs(n, m) = double(0);
        for sim_count=1:number_of_simulations
            try 
                sim.Ms(n,m) = sim.repMax_result{n,m}.dataObj.M;
                current_m = sim.Ms(n,m);
                sim.Ns(n, m) = sim.repMax_result{n,m}.dataObj.N;
                current_n = sim.Ns(n, m);

                
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
                
                RepMax_tasks_workers_costs = rep_max_x_reshaped .* dataObj.all_tasks_workers_costs;
                RepMax_current_total_cost = sum(RepMax_tasks_workers_costs, 'all');
                
                %True
                true_x = sim.true_result{n, m}.all_sims{sim_count}.x;
                true_x_reshaped = reshape(true_x, [current_m, current_n]);
                true_replicas = sum(true_x_reshaped, 2);
                
                TRUE_tasks_workers_costs = true_x_reshaped .* dataObj.all_tasks_workers_costs;
                TRUE_current_total_cost = sum(TRUE_tasks_workers_costs, 'all');
                

                %PBTA
                PBTA_x = sim.PBTA_result{n, m}.all_sims{sim_count}.x;
                PBTA_x_reshaped = reshape(PBTA_x, [current_m, current_n]);
                PBTA_replicas = sum(PBTA_x_reshaped, 2);
                
                PBTA_tasks_workers_costs = PBTA_x_reshaped .* dataObj.all_tasks_workers_costs;
                PBTA_current_total_cost = sum(PBTA_tasks_workers_costs, 'all');
                
                %PBTA MCMF
                PBTA_MCMF_x = double(sim.PBTA_result{n, m}.MCMF_result.X);
                PBTA_MCMF_x_reshaped = reshape(PBTA_MCMF_x, [current_m, current_n]);
                PBTA_MCMF_replicas = sum(PBTA_MCMF_x_reshaped, 2);
                
                PBTA_MCMF_tasks_workers_costs = PBTA_MCMF_x_reshaped .* dataObj.all_tasks_workers_costs;
                PBTA_MCMF_current_total_cost = sum(PBTA_MCMF_tasks_workers_costs, 'all');
                
                %PBTA relibaility and repuation constraints dropped
                PBTA_priority_dropped_x = sim.PBTA_priority_dropped{n, m}.all_sims{sim_count}.x;
                PBTA_priority_dropped_x_reshaped = reshape(PBTA_priority_dropped_x, [current_m, current_n]);
                PBTA_priority_dropped_replicas = sum(PBTA_priority_dropped_x_reshaped, 2);

                PBTA_priority_dropped_tasks_workers_costs = PBTA_priority_dropped_x_reshaped .* dataObj.all_tasks_workers_costs;
                PBTA_priority_dropped_current_total_cost = sum(PBTA_priority_dropped_tasks_workers_costs, 'all');
                
                %sum previous costs with the current
                
                PBTA_average_total_costs(n, m) = PBTA_average_total_costs(n, m) + PBTA_current_total_cost;
                PBTA_MCMF_average_total_costs(n, m) = PBTA_MCMF_average_total_costs(n, m) + PBTA_MCMF_current_total_cost;
                PBTA_priority_dropped_average_total_costs(n, m) = PBTA_priority_dropped_average_total_costs(n, m) + PBTA_priority_dropped_current_total_cost;
                RepMax_average_total_costs(n, m) = RepMax_average_total_costs(n, m) + RepMax_current_total_cost;
                TRUE_average_total_costs(n, m) = TRUE_average_total_costs(n, m) + TRUE_current_total_cost;

            catch ex
                disp('error occured!');
                rethrow(ex)
            end
            %% Step 3: calculate the averages
            %%PBTA
            PBTA_average_total_costs(n, m) = PBTA_average_total_costs(n, m) ./ number_of_simulations;
            %%PBTA MCMF
            PBTA_MCMF_average_total_costs(n, m) = PBTA_MCMF_average_total_costs(n, m) ./ number_of_simulations;
            %%PBTA reliability and reputation dropped
            PBTA_priority_dropped_average_total_costs(n, m) = PBTA_priority_dropped_average_total_costs(n, m) ./ number_of_simulations;
            %%RepMax
            RepMax_average_total_costs(n, m) = RepMax_average_total_costs(n, m) ./ number_of_simulations;
            %%TRUE
            TRUE_average_total_costs(n, m) = TRUE_average_total_costs(n, m) ./ number_of_simulations;
            
            %% Step 4: build graphs
            PBTA_plot = strcat(PBTA_plot, '(', num2str(current_m),', ', num2str(PBTA_average_total_costs(n, m)), ')');
            PBTA_MCMF_plot = strcat(PBTA_MCMF_plot, '(', num2str(current_m),', ', num2str(PBTA_MCMF_average_total_costs(n, m)), ')');
            PBTA_priority_dropped_plot = strcat(PBTA_priority_dropped_plot, '(', num2str(current_m),', ', num2str(PBTA_priority_dropped_average_total_costs(n, m)), ')');
            RepMax_plot = strcat(RepMax_plot, '(', num2str(current_m),', ', num2str(RepMax_average_total_costs(n, m)), ')');
            TRUE_plot = strcat(TRUE_plot, '(', num2str(current_m),', ', num2str(TRUE_average_total_costs(n, m)), ')');
        end%Sim for loop
        
    end%tasks for loop
    disp(strcat('###Task drop rate, for M = ', num2str(current_m), 'and N = ', num2str(current_n), '###'));
    disp('PBTA Plot: ');
    disp(PBTA_plot);
    disp('PBTA MCMF Plot: ');
    disp(PBTA_MCMF_plot);
    disp('PBTA priority dropped Plot: ');
    disp(PBTA_priority_dropped_plot);
    disp('RepMax Plot: ');
    disp(RepMax_plot);
    disp('TRUE Plot: ');
    disp(TRUE_plot);
end%workers for loop

