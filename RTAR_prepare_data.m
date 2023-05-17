function [dataObj] = RTAR_prepare_data(dataObj)
    dataObj.numOfVars = dataObj.N .* dataObj.M;
    if (~isfield(dataObj, 'run_heuristic'))
        dataObj.run_heuristic = true;
    end
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
    if (~isfield(dataObj, 'con_workers_per_task_enabled'))
        dataObj.con_workers_per_task_enabled = true;
    end
    if (~isfield(dataObj, 'MCMF_constraints_enabled'))
        dataObj.MCMF_constraints_enabled = true;
    end
    if (~isfield(dataObj, 'budget_con_enabled'))
        dataObj.budget_con_enabled = true;
    end
    if (~isfield(dataObj, 'fitness_norm_factor_min'))
        dataObj.fitness_norm_factor_min = 1;
    end
    if (~isfield(dataObj, 'fitness_norm_factor_max'))
        dataObj.fitness_norm_factor_max = 100;
    end
    
    if (~isfield(dataObj, 'con_b_enabled_aux'))
        dataObj.con_b_enabled_aux = true;
    end
    if (~isfield(dataObj, 'con_c_enabled_aux'))
        dataObj.con_c_enabled_aux = true;
    end
    if (~isfield(dataObj, 'con_e_enabled_aux'))
        dataObj.con_e_enabled_aux = true;
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
    
    if (~isfield(dataObj, 'tasks_budgets_fromVal'))
        dataObj.tasks_budgets_fromVal = 10;%was 5
    end
    
    if (~isfield(dataObj, 'tasks_budgets_toVal'))
        dataObj.tasks_budgets_toVal = 100;%was 5
    end
    
    if (~isfield(dataObj, "tasks_budgets"))
        dataObj.tasks_budgets = dataObj.tasks_budgets_fromVal + (dataObj.tasks_budgets_toVal - dataObj.tasks_budgets_fromVal) * rand(1, dataObj.M);  % size  = M
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
        dataObj.worker_fitness_fn = @(worker_reputation) worker_reputation;
        
    end
    if (~isfield(dataObj, "worker_fitness_fn_matlab"))
        dataObj.worker_fitness_fn_matlab = @(worker_reputation) worker_reputation .^ 2;
    end
   
    
    %% Objective function
    if (~isfield(dataObj, "obj_func_anonymous_func"))
        dataObj.obj_func_anonymous_func = @(worker_reputation) worker_reputation;
    end
    ctr = 1;
    dataObj.workers_fitness_fns = zeros(1, dataObj.numOfVars);
    dataObj.workers_fitness_fns_matlab = zeros(1, dataObj.numOfVars);
    for i=1:dataObj.N
        for j=1:dataObj.M
            dataObj.workers_fitness_fns(ctr) = dataObj.worker_fitness_fn(dataObj.workers_history_expected_vals(i));
            dataObj.workers_fitness_fns_matlab(ctr) = dataObj.worker_fitness_fn_matlab(dataObj.workers_history_expected_vals(i));

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
            dataObj.objectiveFunction(ctr) = dataObj.obj_func_anonymous_func(dataObj.workers_fitness_fns(ctr));
            dataObj.objectiveFunction_matlab(ctr) = dataObj.obj_func_anonymous_func(dataObj.workers_fitness_fns_matlab(ctr));
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
    dataObj.G_func_tasks_p_level = @(priorities, priority_level) find(priorities == priority_level);
    dataObj.B_func_p_budget = @(alpha, priority_level) alpha .* priority_level;
 
    
    
    
    %% Preparing constraints matrix A 
    dataObj.A = [];
    dataObj.b = [];
    dataObj.operators = [];
    
    
    if (dataObj.con_b_enabled_aux)
        dataObj.con_b_aux = zeros(dataObj.N * dataObj.M, dataObj.numOfVars);
        for i = 1:dataObj.N * dataObj.M
            row = zeros(1, dataObj.numOfVars);
            row(i) = 1 .* dataObj.tasks_total_delays(i);
            dataObj.con_b_aux(i, :) = row;
        end
        dataObj.con_b_bounds_aux = zeros(dataObj.N * dataObj.M, 1);
        ctr = 1;
        for i = 1:dataObj.N
            for j = 1:dataObj.M
                dataObj.con_b_bounds_aux(ctr, :) = dataObj.tasks_deadlines(j);
                ctr = ctr + 1;
            end
        end
       
        dataObj.con_b_size_aux = size(dataObj.con_b_aux, 1); %N * M
    end
    
     %% Constraint (e) t(w_i, \gamma_j)\lambda_{ij} \leq \gamma^{\text{deadline}}_j (e)
    if (dataObj.con_e_enabled_aux)
        dataObj.con_e_aux = zeros(dataObj.N * dataObj.M, dataObj.numOfVars);
        for i = 1:dataObj.N * dataObj.M
            row = zeros(1, dataObj.numOfVars);
            row(i) = 1 .* dataObj.tasks_execution_times(i);
            dataObj.con_e_aux(i, :) = row;
        end
        dataObj.con_e_bounds_aux = zeros(dataObj.N * dataObj.M, 1);
        ctr = 1;
        for i = 1:dataObj.N
            for j = 1:dataObj.M
                dataObj.con_e_bounds_aux(ctr, :) = dataObj.tasks_deadlines(j);
                ctr = ctr + 1;
            end
        end
        
        dataObj.con_e_size_aux = size(dataObj.con_e_aux, 1); %N * M
    end
        
    
    
    %% Constraint (b) \sum^N_{i=1}\lambda_{ij}w^{\text{pay}}_i  \leq  \gamma^{\text{budget}}_j
    if (dataObj.con_b_enabled)
        dataObj.con_b = zeros(dataObj.M, dataObj.numOfVars);
        dataObj.con_b_bounds = zeros(dataObj.M, 1);
        for j = 1:dataObj.M
            row = zeros(1, dataObj.numOfVars);
            for i = 0:dataObj.N - 1
                idx = (i .* dataObj.M) + j;
                row(idx) = dataObj.workers_fitness_costs(i + 1);
            end
            dataObj.con_b(j, :) = row;
            dataObj.con_b_bounds(j) = dataObj.tasks_budgets(j);
        end
        
        
        dataObj.A = [dataObj.A; dataObj.con_b];
        dataObj.b = [dataObj.b; dataObj.con_b_bounds];
        dataObj.operators = [dataObj.operators repmat('<', 1, size(dataObj.con_b, 1))];
        dataObj.con_b_size = size(dataObj.con_b, 1); %N * M
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
    %% Constraint (min_number_of_workers) \sum^{M}_{j=1}  x_{ij} <= w^{\text{tasks}}_i(d)
    if (dataObj.con_workers_per_task_enabled)
        dataObj.con_workers_per_task = zeros(dataObj.M, dataObj.numOfVars);
        for j = 1 : dataObj.M
            row = zeros(1, dataObj.numOfVars);
            for i = 0:dataObj.M:dataObj.N * dataObj.M - 1
                idx = i + j;
                row(idx) = 1;
            end
            dataObj.con_workers_per_task(j, :) = row;
        end
        

        dataObj.A = [dataObj.A; dataObj.con_workers_per_task];
        dataObj.b = [dataObj.b; ones(dataObj.M, 1) .* dataObj.N];
        dataObj.operators = [dataObj.operators repmat('<', 1, size(dataObj.con_workers_per_task, 1))];
        dataObj.con_workers_per_task_size = size(dataObj.con_workers_per_task, 1);% M
        
        dataObj.con_workers_per_task_2 = zeros(dataObj.M, dataObj.numOfVars);
        for j = 1 : dataObj.M
            row = zeros(1, dataObj.numOfVars);
            for i = 0:dataObj.M:dataObj.N * dataObj.M - 1
                idx = i + j;
                row(idx) = -1;
            end
            dataObj.con_workers_per_task_2(j, :) = row;
        end
        

        dataObj.A = [dataObj.A; dataObj.con_workers_per_task_2];
        dataObj.b = [dataObj.b; ones(dataObj.M, 1) .* -1];
        dataObj.operators = [dataObj.operators repmat('<', 1, size(dataObj.con_workers_per_task_2, 1))];
        dataObj.con_workers_per_task_2_size = size(dataObj.con_workers_per_task_2, 1);% M
    end
    
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

