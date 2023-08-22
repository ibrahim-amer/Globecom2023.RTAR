function [dataObj] = RTAR_slice_data(N, M, dataObj)
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
    dataObj.tasks_comm_delays = dataObj.tasks_comm_delays(:, 1:dataObj.numOfVars);
    dataObj.tasks_execution_times = dataObj.tasks_execution_times(:, 1:dataObj.numOfVars);
    dataObj.tasks_priorities = dataObj.tasks_priorities(:, 1:M);
    dataObj.tasks_budgets = dataObj.tasks_budgets(1:M);
    
    dataObj.tasks_deadlines = dataObj.tasks_deadlines(:, 1:M);
    if (isfield(dataObj, 'comp_energies'))
        dataObj.max_energies = dataObj.max_energies(:, 1:N);
    end
    if (isfield(dataObj, 'comp_energies'))
        dataObj.comp_energies = dataObj.comp_energies(1:dataObj.numOfVars, :);
    end

    dataObj = RTAR_prepare_data(dataObj);
    
end

