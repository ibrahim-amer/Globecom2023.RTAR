function [dataObj] = PBTA_prepare_data(dataObj)
    dataObj.numOfVars = dataObj.N .* dataObj.M;
    if (~isfield(dataObj, 'con_b_enabled'))
        dataObj.con_b_enabled = true;
    end
    if (~isfield(dataObj, 'con_c_enabled'))
        dataObj.con_c_enabled = true;
    end
    if (~isfield(dataObj, 'con_d_enabled'))
        dataObj.con_d_enabled = true;
    end
    if (~isfield(dataObj, 'con_e_enabled'))
        dataObj.con_e_enabled = true;
    end
    if (~isfield(dataObj, 'con_f_enabled'))
        dataObj.con_f_enabled = true;
    end
    if (~isfield(dataObj, 'con_2f_enabled'))
        dataObj.con_2f_enabled = true;
    end
    if (~isfield(dataObj, 'con_g_enabled'))
        dataObj.con_g_enabled = true;
    end
    if (~isfield(dataObj, 'MCMF_constraints_enabled'))
        dataObj.MCMF_constraints_enabled = 1;
    end
    if (~isfield(dataObj, 'budget_con_enabled'))
        dataObj.budget_con_enabled = 1;
    end
    if (~isfield(dataObj, 'fitness_norm_factor_min'))
        dataObj.fitness_norm_factor_min = 1;
    end
    if (~isfield(dataObj, 'fitness_norm_factor_max'))
        dataObj.fitness_norm_factor_max = 100;
    end
    %% Problem formulation and system model can be found here: 
    % https://skillful-honesty-f66.notion.site/Meeting-Preparation-March-23-2022-b7f0da29e5694554ba0f07d0acefe679
    %% Preparing Constants: 
    % density: 100MHz - 300 MHz
    % CPU frequency: 1-4 GhZ
    % data size: 1-50 MB
    %%Preparing constants
    %Kappa = 1e-29
    %Worker CPU Frequency
    if (~isstruct(dataObj))
        error('dataObj is not a struct!');
    end
    
    if (~isfield(dataObj, 'max_energy'))
        dataObj.max_energy = 2.1;%was 1
        
    end
    if (~isfield(dataObj, 'max_energies'))
        dataObj.max_energies = dataObj.max_energy .* ones(1, dataObj.N);%size N
    end
    
    %Communication model parameters
    if (~isfield(dataObj, "trans_power"))
        dataObj.trans_power = 50 * 1e-3; %50 mWatt;
    end
    if (~isfield(dataObj, "path_loss_exp"))
        dataObj.path_loss_exp = 2;
    end
    
    if (~isfield(dataObj, "sigma_sq"))
        dataObj.sigma_sq = 1e-11;
    end
    
    if (~isfield(dataObj, "controller_bandwidth"))
        dataObj.controller_bandwidth = 10e6;
    end
    
    dataObj.bandwidth_per_worker = dataObj.controller_bandwidth ./ dataObj.N;
    %Computation model parameters: Kappa 
    if (~isfield(dataObj, "kappa"))
        dataObj.kappa = 1e-29;
    end
    %Reliablity level threshold
    if (~isfield(dataObj, "rel_epsilon"))
        dataObj.rel_epsilon = 0.8;
    end
    if (~isfield(dataObj, "workers_freqs"))
        dataObj.worker_CPU_FromVal = 1e9;
        dataObj.worker_CPU_ToVal = 4e9;
        dataObj.workers_freqs = dataObj.worker_CPU_FromVal + (dataObj.worker_CPU_ToVal - dataObj.worker_CPU_FromVal) * rand(1, dataObj.N);  % size  = N
        %dataObj.workers_freqs = round(dataObj.workers_freqs, 2);
    end
    
    if (~isfield(dataObj, "workers_inferred_cpu_frequencies"))
        dataObj.workers_inferred_cpu_frequencies = zeros(1, dataObj.N);
    end
    
    if (~isfield(dataObj, "workers_channel_states"))
        dataObj.workers_channel_states_from_val = 0.3;
        dataObj.workers_channel_states_to_val = 0.8;
        dataObj.workers_channel_states = dataObj.workers_channel_states_from_val + (dataObj.workers_channel_states_to_val - dataObj.workers_channel_states_from_val) * rand(1, dataObj.N);  % size  = N
    end
    
    
    %Workers utilization
    if (~isfield(dataObj, "workers_utilization"))
       dataObj.worker_utilization_fromval = 0;
       dataObj.worker_utilization_toval = 1;
       dataObj.workers_utilization = dataObj.worker_utilization_fromval + (dataObj.worker_utilization_toval - dataObj.worker_utilization_fromval) * rand(1, dataObj.N);  % size  = N
    end
    
    %Workers maximum number of tasks
    if (~isfield(dataObj, "workers_max_tasks"))
        dataObj.worker_max_tasks_fromval = 1;
        dataObj.worker_max_tasks_toval = 3;
        dataObj.workers_max_tasks = dataObj.worker_max_tasks_fromval + (dataObj.worker_max_tasks_toval - dataObj.worker_max_tasks_fromval) * rand(1, dataObj.N);  % size  = N
        dataObj.workers_max_tasks = round(dataObj.workers_max_tasks);
    end
    %Workers distance from the controller
    if (~isfield(dataObj, "workers_distances"))
        dataObj.worker_distances_fromval = 5;
        dataObj.worker_distances_toval = 50;
        dataObj.workers_distances = dataObj.worker_distances_fromval + (dataObj.worker_distances_toval - dataObj.worker_distances_fromval) * rand(1, dataObj.N);  % size  = N
        %dataObj.workers_distances = round(dataObj.workers_distances);
    end
    %Workers Rayleigh coefficient
    if (~isfield(dataObj, "workers_rayleigh"))
        dataObj.workers_rayleigh = exprnd(1, [1, dataObj.N]); %mu = 1 -->unit mean
    end
    %Workers channel gain
    dataObj.workers_channel_gain = (dataObj.workers_distances .^ -dataObj.path_loss_exp) .* dataObj.workers_rayleigh;
    
    %SNR
    dataObj.SNR = (dataObj.trans_power .* dataObj.workers_channel_gain) ./ dataObj.sigma_sq;
    %Data rate
    dataObj.workers_data_rates = dataObj.bandwidth_per_worker .* log2(1 + dataObj.SNR); 
    
    %Workers hazard rates
     if (~isfield(dataObj, "workers_hazard_rates"))
        dataObj.worker_hazzard_rate_fromval = 0.01;
        dataObj.worker_hazzard_rate_toval = 0.5;
        dataObj.workers_hazard_rates = dataObj.worker_hazzard_rate_fromval + (dataObj.worker_hazzard_rate_toval - dataObj.worker_hazzard_rate_fromval) * rand(1, dataObj.N);  % size  = N
        %dataObj.workers_hazard_rates = round(dataObj.workers_hazard_rates);
     end
    
     %Reliability probability function
     if (~isfield(dataObj, "rel_prop_t"))
         dataObj.rel_prop_t = @(beta, t) exp(-beta .* t);
     end
     
     
    %Workers history
    if (~isfield(dataObj, "workers_history"))
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
        
    end
    
    %Workers cost 
    if (~isfield(dataObj, "workers_costs"))
        dataObj.workers_costs_fromval = 0.5;
        dataObj.workers_costs_toval = 2;
        dataObj.workers_costs = dataObj.workers_costs_fromval + (dataObj.workers_costs_toval - dataObj.workers_costs_fromval) * rand(1, dataObj.N);  % size  = N
    end
    
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%TASK related
    %data%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %Tasks' Processing Density
    if (~isfield(dataObj, "tasks_pdensity"))
        dataObj.task_pdensity_fromVal = 1e2;
        dataObj.task_pdensity_toVal = 5e2;
        dataObj.tasks_pdensity = dataObj.task_pdensity_fromVal + (dataObj.task_pdensity_toVal - dataObj.task_pdensity_fromVal) * rand(1, dataObj.M);  % size  = M
        %dataObj.tasks_pdensity = round(dataObj.tasks_pdensity, 2);
    end
    
    %Tasks data size
    if (~isfield(dataObj, "tasks_dataSize"))
        dataObj.task_dataSize_fromVal = 1e2;
        dataObj.task_dataSize_toVal = 20e2;
        dataObj.tasks_dataSize = dataObj.task_dataSize_fromVal + (dataObj.task_dataSize_toVal - dataObj.task_dataSize_fromVal) * rand(1, dataObj.M);  % size  = M
        %dataObj.tasks_dataSize = round(dataObj.tasks_dataSize, 2);
    end
    
    %Tasks CPU requirement
    if (~isfield(dataObj, "tasks_CPU_req"))
        dataObj.task_CPU_fromVal = 1e9;
        dataObj.task_CPU_toVal = 2e9;
        dataObj.tasks_CPU_req = dataObj.task_CPU_fromVal + (dataObj.task_CPU_toVal - dataObj.task_CPU_fromVal) * rand(1, dataObj.M);  % size  = M
        %dataObj.tasks_CPU_req = round(dataObj.tasks_dataSize, 2);
    end
    
    %Tasks deadlines - uniformly distributed
    if (~isfield(dataObj, "tasks_deadlines"))
        dataObj.task_deadline_fromVal = 8;%was 5
        dataObj.task_deadline_toVal = 10;%was 20
        dataObj.tasks_deadlines = dataObj.task_deadline_fromVal + (dataObj.task_deadline_toVal - dataObj.task_deadline_fromVal) * rand(1, dataObj.M);  % size  = M
        %dataObj.tasks_deadlines = round(dataObj.tasks_deadlines, 2);
    end
    
    if (~isfield(dataObj, 'delay_dividend'))
        dataObj.delay_dividend = 10;
    end
    
    %Computation delays
    tasks_specs = dataObj.tasks_pdensity .* dataObj.tasks_dataSize; % vectorSize = M
    dataObj.tasks_comp_delays = zeros(1, dataObj.numOfVars);
    ctr = 1;
    for i=1:dataObj.N
        dataObj.tasks_comp_delays(ctr:ctr+dataObj.M - 1) = tasks_specs ./ dataObj.workers_freqs(i);
        ctr = ctr + dataObj.M;
    end
    
    %Communication delays
    dataObj.tasks_comm_delays = zeros(1, dataObj.numOfVars);
    ctr = 1;
    for i=1:dataObj.N
        dataObj.tasks_comm_delays(ctr:ctr+dataObj.M - 1) = (dataObj.tasks_dataSize ./ dataObj.workers_data_rates(i));
        ctr = ctr + dataObj.M;
    end
    
    dataObj.tasks_total_delays = (dataObj.tasks_comp_delays + dataObj.tasks_comm_delays) / dataObj.delay_dividend;
    
    %Task execution time on a worker given their utilization
    if (~isfield(dataObj, "tasks_execution_times"))
        dataObj.tasks_execution_times = zeros(1, dataObj.numOfVars);
        ctr = 1;
        for i = 1:dataObj.N
            dataObj.tasks_execution_times(ctr:ctr+dataObj.M - 1) = tasks_specs ./ dataObj.workers_freqs(i) .* dataObj.workers_utilization(i);
            ctr = ctr + dataObj.M;
        end
    end
    
    %Tasks priorities
    if (~isfield(dataObj, "priority_levels"))
        dataObj.priority_levels = 5;
    end
    if (~isfield(dataObj, "tasks_priorities"))
        dataObj.task_priority_fromval = 0.5;
        dataObj.task_priority_toval = dataObj.priority_levels;
        dataObj.tasks_priorities = dataObj.task_priority_fromval + (dataObj.task_priority_toval - dataObj.task_priority_fromval) * rand(1, dataObj.M);  % size  = M
        dataObj.tasks_priorities = round(dataObj.tasks_priorities);
    end
    
    %max_delay = task_dataSize_toVal * task_pdensity_toVal / worker_CPU_FromVal;
    %dataObj.tasks_comp_delays = dataObj.tasks_comp_delays / max_delay;
    
   
    
    %Workers reliability
    dataObj.workers_tasks_rel_prop = zeros(1, dataObj.numOfVars);
    %dataObj.workers_tasks_rel_prop = [];
    ctr = 1;
    for i=1:dataObj.N
        for j=1:dataObj.M
            dataObj.workers_tasks_rel_prop(ctr) = dataObj.rel_prop_t(dataObj.workers_hazard_rates(i), dataObj.tasks_deadlines(j));
            ctr = ctr + 1;
        end
    end
    
     %Modify the values of alphas and betas to match the reliability of
    %the worker. If the worker's reliability is less than epsilon, then
    %the the betas will be incremented for this worker and alphas
    %otherwise.
%     workers_tasks_rel_prob = reshape(dataObj.workers_tasks_rel_prop, [dataObj.M dataObj.N])';
%     workers_tasks_average_prob = sum(workers_tasks_rel_prob, 2) ./ dataObj.M;
%     for i = 1:dataObj.N
%         if (workers_tasks_average_prob(i) > 0.5)
%             dataObj.workers_history_alphas(i) = 1;
%             dataObj.workers_history_betas(i) = 0;
%         else
%             dataObj.workers_history_alphas(i) = 0;
%             dataObj.workers_history_betas(i) = 1;
%         end
%     end
    %Beta distribution expected value per worker
    dataObj.workers_history_expected_vals = zeros(1, dataObj.N);
    dataObj.workers_history_expected_vals = dataObj.workers_history_alphas ./ (dataObj.workers_history_alphas + dataObj.workers_history_betas);
    dataObj.workers_good_rep_cdf = betacdf(0.7, dataObj.workers_history_alphas, dataObj.workers_history_betas, "upper");
    dataObj.workers_bad_rep_cdf = betacdf(0.7, dataObj.workers_history_alphas, dataObj.workers_history_betas);
%     for i = 1:dataObj.N
%         dataObj.workers_history_expected_vals(i) = dataObj.workers_history_alphas(i) ./ (dataObj.workers_history_alphas(i) + dataObj.workers_history_betas(i));
%         dataObj.workers_good_rep_cdf(i) = betacdf(0.7, dataObj.workers_history_alphas(i), dataObj.workers_history_betas(i), "upper");
%         dataObj.workers_bad_rep_cdf(i) = betacdf(0.7, dataObj.workers_history_alphas(i), dataObj.workers_history_betas(i));
%     end
    
    %All workers-tasks costs (used to ease simulations)
    if (~isfield(dataObj, ''))
        dataObj.all_tasks_workers_costs = zeros(dataObj.M, dataObj.N);
        for i = 1:dataObj.N
            for j = 1:dataObj.M
                dataObj.all_tasks_workers_costs(j, i) = dataObj.workers_costs(i) .* dataObj.tasks_dataSize(j) .* dataObj.tasks_pdensity(j);
            end
        end
    end
    
    %Workers fitness function
    if (~isfield(dataObj, "worker_fitness_fn"))
        dataObj.worker_fitness_fn = @(rel_battery, worker_reputation, worker_cost, ... 
                    task_data_size, task_processing_density, worker_cpu, worker_snr) (exp(worker_reputation .^ 2) + worker_cpu + worker_snr);
        
    end
    if (~isfield(dataObj, "worker_fitness_fn_matlab"))
        dataObj.worker_fitness_fn_matlab = @(rel_battery, worker_reputation, worker_cost, ... 
                    task_data_size, task_processing_density, worker_cpu, worker_snr) (exp(worker_reputation .^ 2) + worker_cpu + worker_snr);
    end
    
 
    
    
    %% Find the optimal number of replicas for each task according to the closed-form
    %Solution formula
    [num_of_replicas, actual_allocated_total_replicas] = PBTA_optimal_num_of_replicas_closed_form(dataObj.N, dataObj.M, dataObj.tasks_priorities);
    dataObj.optimal_num_of_replicas_closed_form = num_of_replicas{1};
    dataObj.optimal_total_allocated_replicas_closed_form = actual_allocated_total_replicas;
    
    %% Objective function
    if (~isfield(dataObj, "obj_func_anonymous_func"))
        dataObj.obj_func_anonymous_func = @(worker_fitness_fn, task_priority) worker_fitness_fn .* 1;
    end
    ctr = 1;
    dataObj.workers_fitness_fns = zeros(1, dataObj.numOfVars);
    dataObj.workers_fitness_fns_matlab = zeros(1, dataObj.numOfVars);
    for i=1:dataObj.N
        for j=1:dataObj.M
            if (isfield(dataObj, "workers_inferred_cpu_frequencies"))
                dataObj.workers_fitness_fns(ctr) = dataObj.worker_fitness_fn(dataObj.workers_tasks_rel_prop(ctr), dataObj.workers_history_expected_vals(i), ...
                    dataObj.workers_costs(i), dataObj.tasks_dataSize(j), dataObj.tasks_pdensity(j), dataObj.workers_inferred_cpu_frequencies(i), dataObj.SNR(i));
                dataObj.workers_fitness_fns_matlab(ctr) = dataObj.worker_fitness_fn_matlab(dataObj.workers_tasks_rel_prop(ctr), dataObj.workers_history_expected_vals(i), ...
                    dataObj.workers_costs(i), dataObj.tasks_dataSize(j), dataObj.tasks_pdensity(j), dataObj.workers_inferred_cpu_frequencies(i), dataObj.SNR(i));
            else
                dataObj.workers_fitness_fns(ctr) = dataObj.worker_fitness_fn(dataObj.workers_tasks_rel_prop(ctr), dataObj.workers_history_expected_vals(i), ...
                    dataObj.workers_costs(i), dataObj.tasks_dataSize(j), dataObj.tasks_pdensity(j));
                dataObj.workers_fitness_fns_matlab(ctr) = dataObj.worker_fitness_fn_matlab(dataObj.workers_tasks_rel_prop(ctr), dataObj.workers_history_expected_vals(i), ...
                    dataObj.workers_costs(i), dataObj.tasks_dataSize(j), dataObj.tasks_pdensity(j));
            end
            
            ctr = ctr + 1;
        end
    end
    %dataObj.workers_fitness_fns_inverse = 1 ./ dataObj.workers_fitness_fns;
    dataObj.workers_fitness_fns_reshaped = reshape(dataObj.workers_fitness_fns, [dataObj.N, dataObj.M])';
    %Normalize fitness functions between 0--100
    dataObj.workers_fitness_fns = (normalize(dataObj.workers_fitness_fns, "range") .* (dataObj.fitness_norm_factor_max - dataObj.fitness_norm_factor_min)) + dataObj.fitness_norm_factor_min;
    dataObj.workers_fitness_fns_matlab = (normalize(dataObj.workers_fitness_fns, "range") .* (dataObj.fitness_norm_factor_max - dataObj.fitness_norm_factor_min)) + dataObj.fitness_norm_factor_min;
    dataObj.objectiveFunction = zeros(1, dataObj.numOfVars);
    dataObj.objectiveFunction_matlab = zeros(1, dataObj.numOfVars);
    ctr = 1;
    for i = 1:dataObj.N
        for j = 1:dataObj.M
            dataObj.objectiveFunction(ctr) = dataObj.obj_func_anonymous_func(dataObj.workers_fitness_fns(ctr), dataObj.tasks_priorities(j));
            dataObj.objectiveFunction_matlab(ctr) = dataObj.obj_func_anonymous_func(dataObj.workers_fitness_fns_matlab(ctr), dataObj.tasks_priorities(j));
            ctr = ctr + 1;
        end
    end
    dataObj.objectiveFunction_inv = 1 ./ dataObj.objectiveFunction;
    dataObj.objectiveFunction_inv_matlab = 1 ./ dataObj.objectiveFunction_matlab;
    
    %Assign costs to workers based on their fitness functions    
    dataObj.workers_fitness_costs = zeros(1, dataObj.N);
    
    for w = 1:dataObj.N
        for i = dataObj.fitness_norm_factor_min:20:dataObj.fitness_norm_factor_max
            if (dataObj.workers_fitness_fns(w) >= i && dataObj.workers_fitness_fns(w) < i + 20)
                dataObj.workers_fitness_costs(w) = i + 19;
            end
        end
    end
    if (~isfield(dataObj, "alpha_for_p_budget"))
        dataObj.alpha_for_p_budget = 100;
    end
    dataObj.tasks_budgets = zeros(1, dataObj.M);
    dataObj.G_func_tasks_p_level = @(priorities, priority_level) find(priorities == priority_level);
    dataObj.B_func_p_budget = @(alpha, priority_level) alpha .* priority_level;
    %Tasks budget limits based on their priorities
%     for t = 1:dataObj.M
%         if (dataObj.tasks_priorities(t) == 5)
%             dataObj.tasks_budgets(t) = 500;
%         elseif (dataObj.tasks_priorities(t) == 4)
%             dataObj.tasks_budgets(t) = 400;
%         elseif (dataObj.tasks_priorities(t) == 3)
%             dataObj.tasks_budgets(t) = 400;
%         elseif (dataObj.tasks_priorities(t) == 2)
%             dataObj.tasks_budgets(t) = 200;
%         elseif (dataObj.tasks_priorities(t) == 1)
%             dataObj.tasks_budgets(t) = 100;
%         end
%     end
    
    
    
    %% Preparing constraints matrix A 
    dataObj.A = [];
    dataObj.b = [];
    dataObj.operators = [];
    
        
    
    
    %% Constraint (b) \tau^{\text{comp}}_{ij} x_{ij} <=  t^{\text{deadline}}_j (b)
    if (dataObj.con_b_enabled)
        dataObj.con_b = zeros(dataObj.N * dataObj.M, dataObj.numOfVars);
        for i = 1:dataObj.N * dataObj.M
            row = zeros(1, dataObj.numOfVars);
            row(i) = 1 .* dataObj.tasks_total_delays(i);
            dataObj.con_b(i, :) = row;
        end
        dataObj.con_b_bounds = zeros(dataObj.N * dataObj.M, 1);
        ctr = 1;
        for i = 1:dataObj.N
            for j = 1:dataObj.M
                dataObj.con_b_bounds(ctr, :) = dataObj.tasks_deadlines(j);
                ctr = ctr + 1;
            end
        end
        dataObj.A = [dataObj.A; dataObj.con_b];
        dataObj.b = [dataObj.b; dataObj.con_b_bounds];
        dataObj.operators = [dataObj.operators repmat('<', 1, size(dataObj.con_b, 1))];
        dataObj.con_b_size = size(dataObj.con_b, 1); %N * M
    end
    

    %% Constraint (c)  \sum^M_{j = 1}x_{ij} E^{\text{comp}}_{ij} \leq E^{\text{max}}_i
    if (dataObj.con_c_enabled)
        
        energy_task_specs_tmp = dataObj.kappa .* dataObj.tasks_dataSize .* dataObj.tasks_pdensity; %size M
        dataObj.comp_energies = zeros(dataObj.N * dataObj.M, 1); %size M * N, 1
        ctr = 1;
        for i = 1:dataObj.N
            for j = 1:dataObj.M
                dataObj.comp_energies(ctr) = energy_task_specs_tmp(j) .* dataObj.workers_freqs(i)^2;
                ctr = ctr + 1;
            end
        end
        dataObj.con_c = zeros(dataObj.N, dataObj.numOfVars);
        ctr = 1;
        for i = 0:dataObj.M:dataObj.N * dataObj.M - 1
            row = zeros(1, dataObj.numOfVars);
            for j = 1:dataObj.M
                row(1, i+j) = dataObj.comp_energies(i+j);            
            end
            dataObj.con_c(ctr, :) = row;
            ctr = ctr + 1;
        end

        dataObj.con_c_bounds = zeros(dataObj.N, 1);
        ctr = 1;
        for i = 1:dataObj.N
            dataObj.con_c_bounds(ctr, :) = dataObj.max_energies(i);
            ctr = ctr + 1;
        end

        dataObj.A = [dataObj.A; dataObj.con_c];
        dataObj.b = [dataObj.b;  dataObj.con_c_bounds];
        dataObj.operators = [dataObj.operators repmat('<', 1, size(dataObj.con_c, 1))];
        dataObj.con_c_size = size(dataObj.con_c, 1); %N * M
    end
    
    %% Constraint (d) \sum^{M}_{j=1}  x_{ij} <= w^{\text{tasks}}_i(d)
    if (dataObj.con_d_enabled)
        dataObj.con_d = zeros(dataObj.N, dataObj.numOfVars);
        ctr = 1;
        for i = 0:dataObj.M:dataObj.N * dataObj.M - 1
            row = zeros(1, dataObj.numOfVars);
            for j = 1:dataObj.M
                row(1, i+j) = 1;            
            end
            dataObj.con_d(ctr, :) = row;
            ctr = ctr + 1;
        end

        dataObj.A = [dataObj.A; dataObj.con_d];
        dataObj.b = [dataObj.b; dataObj.workers_max_tasks'];
        dataObj.operators = [dataObj.operators repmat('<', 1, size(dataObj.con_d, 1))];
        dataObj.con_d_size = size(dataObj.con_d, 1);% N
    end
    
    %% Constraint (e) t(w_i, \gamma_j)\lambda_{ij} \leq \gamma^{\text{deadline}}_j (e)
    if (dataObj.con_e_enabled)
        dataObj.con_e = zeros(dataObj.N * dataObj.M, dataObj.numOfVars);
        for i = 1:dataObj.N * dataObj.M
            row = zeros(1, dataObj.numOfVars);
            row(i) = 1 .* dataObj.tasks_execution_times(i);
            dataObj.con_e(i, :) = row;
        end
        dataObj.con_e_bounds = zeros(dataObj.N * dataObj.M, 1);
        ctr = 1;
        for i = 1:dataObj.N
            for j = 1:dataObj.M
                dataObj.con_e_bounds(ctr, :) = dataObj.tasks_deadlines(j);
                ctr = ctr + 1;
            end
        end
        dataObj.A = [dataObj.A; dataObj.con_e];
        dataObj.b = [dataObj.b; dataObj.con_e_bounds];
        dataObj.operators = [dataObj.operators repmat('<', 1, size(dataObj.con_e, 1))];
        dataObj.con_e_size = size(dataObj.con_e, 1); %N * M
    end
    
    %% Constraint (f) \sum^{M}_{j=1}\lambda_{ij} - \frac{N}{M}\gamma^{\text{priority}}_j \leq 0
    if (dataObj.con_f_enabled)
        dataObj.con_f = zeros(dataObj.M, dataObj.numOfVars);
        for j = 1:dataObj.M
            row = zeros(1, dataObj.numOfVars);
            for i = 0:dataObj.N - 1
                idx = (i .* dataObj.M) + j;
                row(1, idx) = 1;
            end
            dataObj.con_f(j, :) = row;
        end

        dataObj.con_f_bounds = zeros(dataObj.M, 1);
        for j = 1:dataObj.M
            dataObj.con_f_bounds(j, :) = (dataObj.N ./ dataObj.M) .* dataObj.tasks_priorities(j);
        end
        dataObj.A = [dataObj.A; dataObj.con_f];
        dataObj.b = [dataObj.b; dataObj.con_f_bounds];
        dataObj.operators = [dataObj.operators repmat('<', 1, size(dataObj.con_f, 1))];
        dataObj.con_f_size = size(dataObj.con_f, 1); %N * M
    end
    
    %% Constraint (2f) \sum^{M}_{j=1}\lambda_{ij} - \frac{N}{M}\gamma^{\text{priority}}_j \leq 0
    if (dataObj.con_2f_enabled)
        dataObj.con_2f = zeros(dataObj.M, dataObj.numOfVars);
        for j = 1:dataObj.M
            row = zeros(1, dataObj.numOfVars);
            for i = 0:dataObj.N - 1
                idx = (i .* dataObj.M) + j;
                row(1, idx) = -1;
            end
            dataObj.con_2f(j, :) = row;
        end

        dataObj.con_2f_bounds = zeros(dataObj.M, 1);
        dataObj.A = [dataObj.A; dataObj.con_2f];
        dataObj.b = [dataObj.b; dataObj.con_2f_bounds];
        dataObj.operators = [dataObj.operators repmat('<', 1, size(dataObj.con_2f, 1))];
        dataObj.con_2f_size = size(dataObj.con_2f, 1); %N * M
    end
    
    %% Constraint (g) \sum^{N}_{i=1} \sum^{M}_{j=1}\lambda_{ij} \geq\frac{N}{M}
    if (dataObj.con_g_enabled)
        dataObj.con_g = ones(1, dataObj.numOfVars);
        dataObj.con_g_bounds = sum(dataObj.workers_max_tasks, 'all');
        dataObj.A = [dataObj.A; dataObj.con_g];
        dataObj.b = [dataObj.b; dataObj.con_g_bounds];
        dataObj.operators = [dataObj.operators repmat('<', 1, size(dataObj.con_g, 1))];
        dataObj.con_g_size = size(dataObj.con_g, 1); %N * M

        dataObj.con_2g = ones(1, dataObj.numOfVars) .* -1;
        dataObj.con_2g_bounds = 0;
        dataObj.A = [dataObj.A; dataObj.con_2g];
        dataObj.b = [dataObj.b; dataObj.con_2g_bounds];
        dataObj.operators = [dataObj.operators repmat('<', 1, size(dataObj.con_2g, 1))];
        dataObj.con_2g_size = size(dataObj.con_2g, 1); %N * M
    end
    
    %% Budget constraint: \sum^N_{i=1}\lambda_{ij}\;Q(f(w_i))\leq L( \gamma^{\text{priority}}_j)
    if (dataObj.budget_con_enabled)
        dataObj.budget_con = zeros(dataObj.priority_levels, dataObj.numOfVars);
        dataObj.budget_con_bounds = zeros(dataObj.priority_levels, 1);
        for k=1:dataObj.priority_levels
            %If there is no task with priority level 'k' in the tasks'
            %list, then continue
            if (length(find(dataObj.tasks_priorities == k)) == 0)
                continue;
            end
            row = zeros(1, dataObj.numOfVars);
            tasks_p_k_level = dataObj.G_func_tasks_p_level(dataObj.tasks_priorities, k);
            
            for j = 1:length(tasks_p_k_level)
                for i = 0:dataObj.N - 1
                    idx = (i .* dataObj.M) + tasks_p_k_level(j);
                    row(1, idx) = dataObj.workers_fitness_costs(i + 1);
                end
            end
            dataObj.budget_con(k, :) = row;
            dataObj.budget_con_bounds(k) = dataObj.B_func_p_budget(dataObj.alpha_for_p_budget, k);
%             dataObj.tasks_budgets(tasks_p_k_level) = dataObj.B_func_p_budget(dataObj.alpha_for_p_budget, k)...
%                                                         ./ length(tasks_p_k_level);
%             if (dataObj.tasks_budgets(tasks_p_k_level) < min(dataObj.workers_fitness_costs))
%                 num_of_tasks_with_budget_equal_min_cost = ceil(dataObj.B_func_p_budget(dataObj.alpha_for_p_budget, k)...
%                                                        ./ min(dataObj.workers_fitness_costs));
%                 dataObj.tasks_budgets(tasks_p_k_level(1:num_of_tasks_with_budget_equal_min_cost)) = ...
%                                                         min(dataObj.workers_fitness_costs);
%                 dataObj.tasks_budgets(tasks_p_k_level(num_of_tasks_with_budget_equal_min_cost + 1:end)) = 0;                                   
%             end
        end
               
        dataObj.A = [dataObj.A; dataObj.budget_con];
        dataObj.b = [dataObj.b; dataObj.budget_con_bounds];
        dataObj.operators = [dataObj.operators repmat('<', 1, size(dataObj.budget_con, 1))];
        dataObj.budget_con_size = size(dataObj.budget_con, 1); %N * M
        
        sorted_workers_costs = sort(dataObj.workers_fitness_costs);
        
        for k=dataObj.priority_levels:-1:1
            p_budget = dataObj.B_func_p_budget(dataObj.alpha_for_p_budget, k);
            tasks_p_k_level = dataObj.G_func_tasks_p_level(dataObj.tasks_priorities, k);
            
            current_w_idx = 1;
            dataObj.tasks_budgets(tasks_p_k_level) = p_budget;
%             while p_budget > 0 && ~isempty(tasks_p_k_level) ...
%                     && current_w_idx < length(sorted_workers_costs)
%                 for t = 1:length(tasks_p_k_level)
%                     if (current_w_idx > length(sorted_workers_costs))
%                         break;
%                     end
%                     if (p_budget < sorted_workers_costs(current_w_idx) || p_budget <= 0)
%                         dataObj.tasks_budgets(tasks_p_k_level(t)) = 0;
%                     else
%                         dataObj.tasks_budgets(tasks_p_k_level(t)) = dataObj.tasks_budgets(tasks_p_k_level(t)) + sorted_workers_costs(current_w_idx);
%                         p_budget = p_budget - sorted_workers_costs(current_w_idx);
%                     end
% 
%                     current_w_idx = current_w_idx + 1;
% %                     current_w_idx = mod(current_w_idx, dataObj.N + 1);
% %                     if (current_w_idx == 0)
% %                         current_w_idx = 1;
% %                     end
%                 end
%             end
            
        end%priority level for
    end%constraint's if
    
    %% Constraint (h) RHS
    
    dataObj.ub = zeros(1, dataObj.N * dataObj.M);
    dataObj.lb = zeros(1, dataObj.N * dataObj.M);
    dataObj.ub(1, 1:dataObj.N * dataObj.M) = 1;
    dataObj.lb(1, 1:dataObj.N * dataObj.M) = 0; %already satisfied
    
    
    %Upper and lower bounds
    %lambdas upper bounds
    lambdasBounds = diag([ones(dataObj.N .* dataObj.M, 1);]);
    lambdasBounds = lambdasBounds(1:dataObj.N * dataObj.M, :);
    dataObj.A = [dataObj.A; lambdasBounds];
    dataObj.b = [dataObj.b; dataObj.ub(1, 1:dataObj.N * dataObj.M)'];
    dataObj.operators = [dataObj.operators repmat('<', 1, dataObj.N * dataObj.M)];
    %lambdas lower bounds
    dataObj.A = [dataObj.A; -1 .* lambdasBounds];
    dataObj.b = [dataObj.b; dataObj.lb(1, 1:dataObj.N * dataObj.M)'];
    dataObj.operators = [dataObj.operators repmat('<', 1, dataObj.N * dataObj.M)];
    dataObj.bounds_size = size(2 * lambdasBounds, 1); % N * M * 2
    
    
end

