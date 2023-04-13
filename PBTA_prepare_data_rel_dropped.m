function [dataObj] = TREED_battery_aware_prepare_data_rel_dropped(dataObj)
    dataObj.numOfVars = dataObj.N .* dataObj.M;
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
    
    %Communication model parameters
    if (~isfield(dataObj, "trans_power"))
        dataObj.trans_power = 50 * 1e-3 %50 mWatt;
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
     
          
    
    %Tasks' Processing Density
    if (~isfield(dataObj, "tasks_pdensity"))
        dataObj.task_pdensity_fromVal = 1e2;
        dataObj.task_pdensity_toVal = 5e2;
        dataObj.tasks_pdensity = dataObj.task_pdensity_fromVal + (dataObj.task_pdensity_toVal - dataObj.task_pdensity_fromVal) * rand(1, dataObj.M);  % size  = M
        %dataObj.tasks_pdensity = round(dataObj.tasks_pdensity, 2);
    end
    
    %Tasks data size
    if (~isfield(dataObj, "tasks_dataSize"))
        dataObj.task_dataSize_fromVal = 1e6;
        dataObj.task_dataSize_toVal = 20e6;
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
    
    %max_delay = task_dataSize_toVal * task_pdensity_toVal / worker_CPU_FromVal;
    %dataObj.tasks_comp_delays = dataObj.tasks_comp_delays / max_delay;
    
    %Workers reliability
    dataObj.workers_tasks_rel_prop = zeros(1, dataObj.numOfVars);
    dataObj.workers_tasks_rel_prop = [];
    ctr = 1;
    for i=1:dataObj.N
        for j=1:dataObj.M
            dataObj.workers_tasks_rel_prop(ctr) = dataObj.rel_prop_t(dataObj.workers_hazard_rates(i), dataObj.tasks_deadlines(j));
            ctr = ctr + 1;
        end
    end
    
    %% Objective function
    dataObj.workers_tasks_diff = zeros(1, dataObj.numOfVars);
    ctr = 1;
    for i = 1:dataObj.N
        for j = 1:dataObj.M
            dataObj.workers_tasks_diff(ctr) = dataObj.workers_freqs(i) - dataObj.tasks_CPU_req(j);
            ctr = ctr + 1;
        end
    end
    dataObj.objectiveFunction = dataObj.workers_tasks_diff;
    
    
    %% Preparing constraints matrix A 
    dataObj.A = [];
    dataObj.b = [];
    dataObj.operators = [];
    %% Constraint w^{\text{CPU}}_i x_{ij} >= t^{\text{CPU}}_j (b)
    con_b = zeros(dataObj.N * dataObj.M, dataObj.numOfVars);
    con_b_bounds = zeros(dataObj.N * dataObj.M, 1);
    ctr = 1;
    for i = 0:dataObj.N - 1
        for j = 1:dataObj.M
            row = zeros(1, dataObj.numOfVars);
            row(((i * dataObj.M) + j)) = dataObj.workers_tasks_diff(ctr);%1 .* dataObj.workers_freqs(i + 1);
            con_b(ctr, :) = row;
            con_b_bounds(ctr, :) = 0;
            ctr = ctr + 1;
        end
    end
    dataObj.A = [dataObj.A; con_b];
    dataObj.b = [dataObj.b; con_b_bounds];
    dataObj.operators = [dataObj.operators repmat('>', 1, size(con_b, 1))];
    dataObj.con_b_size = size(con_b, 1); %N * M
    cond = (dataObj.objectiveFunction' == diag(dataObj.A(1:dataObj.con_b_size, :)));
    assert(sum(cond) == length(cond), 'prepare_data: obj func not equal to constrain (b)');
    
    %% Constraint (c)  \sum^M_{j = 1}x_{ij} E^{\text{comp}}_{ij} \leq E^{\text{max}}_i
    dataObj.max_energy = 3;%was 1
    dataObj.max_energies = dataObj.max_energy .* ones(1, dataObj.N);%size N
    energy_task_specs_tmp = dataObj.kappa .* dataObj.tasks_dataSize .* dataObj.tasks_pdensity; %size M
    dataObj.comp_energies = zeros(dataObj.N * dataObj.M, 1); %size M * N, 1
    ctr = 1;
    for i = 1:dataObj.N
        for j = 1:dataObj.M
            dataObj.comp_energies(ctr) = energy_task_specs_tmp(j) .* dataObj.workers_freqs(i)^2;
            ctr = ctr + 1;
        end
    end
    con_c = zeros(dataObj.N, dataObj.numOfVars);
    ctr = 1;
    for i = 0:dataObj.M:dataObj.N * dataObj.M - 1
        row = zeros(1, dataObj.numOfVars);
        for j = 1:dataObj.M
            row(1, i+j) = dataObj.comp_energies(i+j);            
        end
        con_c(ctr, :) = row;
        ctr = ctr + 1;
    end
    
    con_c_bounds = zeros(dataObj.N, 1);
    ctr = 1;
    for i = 1:dataObj.N
        con_c_bounds(ctr, :) = dataObj.max_energies(i);
        ctr = ctr + 1;
    end
    
    dataObj.A = [dataObj.A; con_c];
    dataObj.b = [dataObj.b;  con_c_bounds];
    dataObj.operators = [dataObj.operators repmat('<', 1, size(con_c, 1))];
    dataObj.con_c_size = size(con_c, 1); %N * M
    

    
    %% Constraint (e) \tau^{\text{comp}}_{ij} x_{ij} <=  t^{\text{deadline}}_j (e)
    con_e = zeros(dataObj.N * dataObj.M, dataObj.numOfVars);
    for i = 1:dataObj.N * dataObj.M
        row = zeros(1, dataObj.numOfVars);
        row(i) = 1 .* dataObj.tasks_total_delays(i);
        con_e(i, :) = row;
    end
    con_e_bounds = zeros(dataObj.N * dataObj.M, 1);
    ctr = 1;
    for i = 1:dataObj.N
        for j = 1:dataObj.M
            con_e_bounds(ctr, :) = dataObj.tasks_deadlines(j);
            ctr = ctr + 1;
        end
    end
    dataObj.A = [dataObj.A; con_e];
    dataObj.b = [dataObj.b; con_e_bounds];
    dataObj.operators = [dataObj.operators repmat('<', 1, size(con_e, 1))];
    dataObj.con_e_size = size(con_e, 1); %N * M
    
    %% Constraint \sum^N_{i = 1} x_{ij} >= 1 (f)
    con_f = zeros(dataObj.M, dataObj.numOfVars);
    for j = 1:dataObj.M
        row = zeros(1, dataObj.numOfVars);
        for ij = 0:dataObj.M:dataObj.N * dataObj.M - 1
            row(j + ij) = 1; 
        end
        con_f(j, :) = row;
    end
    dataObj.A = [dataObj.A; con_f];
    dataObj.b = [dataObj.b; ones(size(con_f, 1), 1)];
    dataObj.operators = [dataObj.operators repmat('>', 1, size(con_f, 1))];
    dataObj.con_f_size = size(con_f, 1);% M
    
    %% Constraint \sum^{M}_{j=1}  x_{ij} <= w^{\text{tasks}}_i(g)
    con_g = zeros(dataObj.N, dataObj.numOfVars);
    ctr = 1;
    for i = 0:dataObj.M:dataObj.N * dataObj.M - 1
        row = zeros(1, dataObj.numOfVars);
        for j = 1:dataObj.M
            row(1, i+j) = 1;            
        end
        con_g(ctr, :) = row;
        ctr = ctr + 1;
    end
    
    dataObj.A = [dataObj.A; con_g];
    dataObj.b = [dataObj.b; dataObj.workers_max_tasks'];
    dataObj.operators = [dataObj.operators repmat('<', 1, size(con_g, 1))];
    dataObj.con_g_size = size(con_g, 1);% N
    

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
    dataObj.A = [dataObj.A; lambdasBounds];
    dataObj.b = [dataObj.b; dataObj.lb(1, 1:dataObj.N * dataObj.M)'];
    dataObj.operators = [dataObj.operators repmat('>', 1, dataObj.N * dataObj.M)];
    dataObj.bounds_size = size(2 * lambdasBounds, 1); % N * M * 2
    
    
end

