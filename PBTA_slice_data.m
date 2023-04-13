function [dataObj] = PBTA_slice_data(N, M, dataObj)
    dataObj.N = N;
    dataObj.M = M;
    dataObj.numOfVars = dataObj.N * dataObj.M;
    dataObj.workers_freqs = dataObj.workers_freqs(1, 1:N);
    
    dataObj.workers_utilization = dataObj.workers_utilization(1, 1:N);
    dataObj.workers_history = dataObj.workers_history(1, 1:N);
    dataObj.workers_history_alphas = dataObj.workers_history_alphas(1, 1:N);
    dataObj.workers_history_betas = dataObj.workers_history_betas(1, 1:N);
    dataObj.workers_history_expected_vals = dataObj.workers_history_expected_vals(1, 1:N);
    dataObj.workers_good_rep_cdf = dataObj.workers_good_rep_cdf(1, 1:N);
    dataObj.workers_bad_rep_cdf = dataObj.workers_bad_rep_cdf(1, 1:N);
    dataObj.all_tasks_workers_costs = dataObj.all_tasks_workers_costs(1:M, 1:N);

    dataObj.workers_costs = dataObj.workers_costs(1, 1:N);
    
    dataObj.workers_max_tasks = dataObj.workers_max_tasks(1, 1:N);
    dataObj.workers_distances = dataObj.workers_distances(1, 1:N);
    dataObj.workers_rayleigh = dataObj.workers_rayleigh(1, 1:N);
    dataObj.workers_channel_gain = dataObj.workers_channel_gain(1, 1:N);
    dataObj.SNR = dataObj.SNR(:, 1:N);
    dataObj.workers_data_rates = dataObj.workers_data_rates(1, 1:N);
    dataObj.workers_hazard_rates = dataObj.workers_hazard_rates(1, 1:N);
    dataObj.tasks_pdensity = dataObj.tasks_pdensity(:, 1:M);
    dataObj.tasks_dataSize = dataObj.tasks_dataSize(:, 1:M);
    dataObj.tasks_comp_delays = dataObj.tasks_comp_delays(:, 1:dataObj.numOfVars);
    
    dataObj.tasks_execution_times = dataObj.tasks_execution_times(:, 1:dataObj.numOfVars);
    dataObj.tasks_priorities = dataObj.tasks_priorities(:, 1:M);
    
    dataObj.tasks_deadlines = dataObj.tasks_deadlines(:, 1:M);
    if (isfield(dataObj, 'comp_energies'))
        dataObj.max_energies = dataObj.max_energies(:, 1:N);
    end
    if (isfield(dataObj, 'comp_energies'))
        dataObj.comp_energies = dataObj.comp_energies(1:dataObj.numOfVars, :);
    end
        
    %            a     b    c   d   e   f_ub  f_lb  g_ub   g_lb
%     dataObj.con_b_size = N*M;
%     dataObj.con_c_size = N;
%     dataObj.con_d_size = M;
%     dataObj.con_e_size = N*M;
%     dataObj.con_f_size = M;
%     dataObj.con_g_size = N;
%     dataObj.con_bounds_size = 2 * (N*M);
% %     
%     dataObj.A_b_n_rows = dataObj.con_b_size + dataObj.con_c_size + dataObj.con_d_size + dataObj.con_e_size + dataObj.con_f_size + dataObj.con_g_size + dataObj.con_bounds_size;
%     dataObj.A = dataObj.A(1:A_b_n_rows, 1:dataObj.numOfVars);
%     dataObj.b = dataObj.b(1:A_b_n_rows, 1);
%     dataObj.ub = dataObj.ub(1, 1:dataObj.numOfVars);
%     dataObj.lb = dataObj.lb(1, 1:dataObj.numOfVars);
%     dataObj.operators = dataObj.operators(:, 1:A_b_n_rows);
%     dataObj.workers_tasks_diff = zeros(1, dataObj.numOfVars);
%     ctr = 1;
%     for i = 1:N
%         for j = 1:M
%             dataObj.workers_tasks_diff(ctr) = dataObj.workers_freqs(i) - dataObj.tasks_CPU_req(j);
%             ctr = ctr + 1;
%         end
%     end
    dataObj = PBTA_prepare_data(dataObj);
    %cond = (dataObj.objectiveFunction' == diag(dataObj.A(1:dataObj.con_b_size, :)));
    %assert(sum(cond) == length(cond), 'prepare_data: obj func not equal to constrain (b)');
end

