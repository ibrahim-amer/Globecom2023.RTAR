%% This script renders a graph that compares PBTA, MCMF, RepMax, TRUE, PBTA-REPD and MCMF-REPD while varying the reputation of the workers and the number of tasks.
%Preparing the data
N_min = 100;
N_max = 100;
N_stepSize = 5;

M_min = 20;
M_max = 50;
M_stepSize = 5;
epochs = 10;



number_of_simulations = 1;
checkConstraints = true;
enable_true_benchmark = false;
N = ceil((N_max - N_min + 1) ./ N_stepSize);
M = ceil((M_max - M_min + 1) ./ M_stepSize);

n_vector = N_min:N_stepSize:N_max;
m_vector = M_min:M_stepSize:M_max;
failure_percentage = 0.5;
 

hazard_rates = 0.2:0.1:1;
H = length(hazard_rates);
all_results = cell(N, M, H);
for h=1:H
    for n=1:N
        for m=1:M
            all_results{n, m, h} = struct();
            workers_channel_states_from_val = 0.3;
            workers_channel_states_to_val = 0.8;

            current_dataObj = RTAR_init_dataObj_sim(n_vector(n), m_vector(m));
            current_dataObj.workers_inferred_cpu_frequencies = ones(1, n_vector(n)) .* 1e9;
            current_dataObj.workers_channel_states = workers_channel_states_from_val + (workers_channel_states_to_val - workers_channel_states_from_val) * rand(1, current_dataObj.N);  % size  = N
            current_MCMF_dataObj = current_dataObj;
            
            current_dataObj.worker_hazzard_rate_toval = hazard_rates(h);
            current_dataObj.workers_hazard_rates = current_dataObj.worker_hazzard_rate_fromval + (current_dataObj.worker_hazzard_rate_toval - current_dataObj.worker_hazzard_rate_fromval) * rand(1, current_dataObj.N);  % size  = N


            %RepMax
            all_results{n, m, h}.repMax_result = RepMax_simulation(n_vector(n), n_vector(n), N_stepSize, m_vector(m), ...
           m_vector(m), M_stepSize, number_of_simulations, current_dataObj, checkConstraints);

            %TRUE
            if (enable_true_benchmark)
                all_results{n, m, h}.true_result = TREED_battery_aware_simulation(n_vector(n), n_vector(n), N_stepSize, m_vector(m), m_vector(m),...
                    M_stepSize, number_of_simulations, current_dataObj, checkConstraints);
            end



            for epoch=1:epochs
                %% ###################################################RTAR and RTAR_H: reputation only with no constraints ###################################################
                %Disabling constraints non-related to reputation

                %Change CPU loads
                current_dataObj.worker_utilization_fromval = 0;
                current_dataObj.worker_utilization_toval = 1;
                current_dataObj.workers_utilization = current_dataObj.worker_utilization_fromval + (current_dataObj.worker_utilization_toval - current_dataObj.worker_utilization_fromval) * rand(1, current_dataObj.N);  % size  = N
                current_dataObj.workers_freqs = current_dataObj.workers_freqs - (current_dataObj.workers_freqs .* current_dataObj.workers_utilization);
                %Change channels conditions
    %             current_dataObj.worker_distances_fromval = 5;
    %             current_dataObj.worker_distances_toval = 50;
    %             current_dataObj.workers_distances = current_dataObj.worker_distances_fromval + (current_dataObj.worker_distances_toval - current_dataObj.worker_distances_fromval) * rand(1, current_dataObj.N);  % size  = N
    %             %Workers channel gain
    %             current_dataObj.workers_channel_gain = (current_dataObj.workers_distances .^ -current_dataObj.path_loss_exp) .* current_dataObj.workers_rayleigh;
    %     
    %             %SNR
    %             current_dataObj.SNR = (current_dataObj.trans_power .* current_dataObj.workers_channel_gain) ./ current_dataObj.sigma_sq;
    %             
                current_dataObj.run_run_heuristic = 1;
                all_results{n, m, h}.RTAR_result = RTAR_simulation(n_vector(n), n_vector(n), N_stepSize, m_vector(m), ... 
                    m_vector(m), M_stepSize, number_of_simulations, current_dataObj, checkConstraints);
    %             current_MCMF_dataObj.run_MCMF = true;
    %             current_MCMF_dataObj.MCMF_constraints_enabled = 0;
    %             all_results{n, m, h}.PBTA_MCMF_result = PBTA_simulation(n_vector(n), n_vector(n), N_stepSize, m_vector(m), ... 
    %                 m_vector(m), M_stepSize, number_of_simulations, current_MCMF_dataObj, checkConstraints);

                %Check the status of the tasks after the recruitment by checking the
                %constraints
                current_m = all_results{n, m, h}.RTAR_result{1,1}.dataObj.M;
                current_n = all_results{n, m, h}.RTAR_result{1,1}.dataObj.N;
                current_dataObj = all_results{n, m, h}.RTAR_result{1, 1}.dataObj;
                %current_MCMF_dataObj = all_results{n, m, h}.PBTA_MCMF_result{1, 1}.dataObj;
                RTAR_workers_history_current_vals = zeros(1, current_n);
                RTAR_H_workers_history_current_vals = zeros(1, current_n);
                workers_avg_inferred_cpu = ones(1, current_n) .* 1e9;
                for sim_count = 1:number_of_simulations
                    %RTAR
                    RTAR_x = all_results{n, m, h}.RTAR_result{1, 1}.all_sims{sim_count}.optimal_solution.x;
                    RTAR_x_reshaped = reshape(RTAR_x, [current_m, current_n]);
                    RTAR_x_n_m_reshaped = RTAR_x_reshaped';
                    RTAR_x_num_of_tasks_per_worker = sum(RTAR_x_n_m_reshaped, 2);
                    RTAR_replicas = sum(RTAR_x_reshaped, 2);


                    %RTAR H
                    RTAR_H_x = all_results{n, m, h}.RTAR_result{1, 1}.all_sims{sim_count}.RTAR_H.X;
                    RTAR_H_x_reshaped = reshape(RTAR_H_x, [current_m, current_n]);
                    RTAR_H_x_n_m_reshaped = RTAR_H_x_reshaped';
                    RTAR_H_x_num_of_tasks_per_worker = sum(RTAR_H_x_n_m_reshaped, 2);
                    RTAR_H_replicas = sum(RTAR_H_x_reshaped, 2);

                    %Find failure probability
                    failure_rel_probs = 1 - current_dataObj.workers_tasks_rel_prop;
                    failure_rel_probs = reshape(failure_rel_probs, [current_m, current_n]);
                    workers_bad_rep_cdf = reshape(current_dataObj.workers_history_expected_vals, [1, current_n]);
                    failure_rep_probs = ones(current_m, current_n) .* (1 - workers_bad_rep_cdf);
                    joint_failure_probs = failure_rel_probs .* failure_rep_probs;
                    failed_nodes_global = joint_failure_probs > failure_percentage;

                    RTAR_tasks_fail_prob = failed_nodes_global;
                    RTAR_workers_failed = RTAR_tasks_fail_prob';
                    RTAR_workers_failed = sum(RTAR_workers_failed, 2);
                    RTAR_actual_workers = RTAR_x_num_of_tasks_per_worker - RTAR_workers_failed;
                    RTAR_actual_workers(RTAR_actual_workers < 0) = 0;
                    RTAR_workers_failure_status = RTAR_actual_workers == 0;
                    RTAR_nodes_failed = sum(RTAR_tasks_fail_prob, 2);
                    RTAR_actual_nodes = abs(RTAR_nodes_failed - RTAR_replicas);
                    RTAR_status = RTAR_actual_nodes == 0;
                    RTAR_percentage = sum(RTAR_status) ./ current_m;


                    %%RTAR H

                    RTAR_H_tasks_fail_prob = failed_nodes_global;
                    RTAR_H_workers_failed = RTAR_H_tasks_fail_prob';
                    RTAR_H_workers_failed = sum(RTAR_H_workers_failed, 2);
                    RTAR_H_actual_workers = RTAR_H_x_num_of_tasks_per_worker - RTAR_H_workers_failed;
                    RTAR_H_actual_workers(RTAR_H_actual_workers < 0) = 0;
                    RTAR_H_workers_failure_status = RTAR_H_actual_workers == 0;
                    RTAR_H_nodes_failed = sum(RTAR_H_tasks_fail_prob, 2);
                    RTAR_H_actual_nodes = abs(RTAR_H_nodes_failed - RTAR_H_replicas);
                    RTAR_H_status = RTAR_H_actual_nodes == 0;
                    RTAR_H_percentage = sum(RTAR_H_status) ./ current_m;



                    %Check constraints


                    RTAR_combined_constraints = [];
                    RTAR_H_combined_constraints = [];

                    %dataObj_temp = RTAR_prepare_data(current_dataObj);%re-prepare the data with constraints
                    %MCMF_dataObj_temp = PBTA_prepare_data(current_MCMF_dataObj);%re-prepare the data with constraints
                    %Constraint b
                    if (current_dataObj.con_b_enabled_aux)
                        con_b_result = current_dataObj.con_b_aux * RTAR_x <= current_dataObj.con_b_bounds_aux;
                        con_b_n_m_reshaped = reshape(con_b_result, [current_m, current_n])';
                        con_b_failed_nodes = sum(con_b_n_m_reshaped == 0, 2) > 0;
                        con_b_non_failed_nodes = ~con_b_failed_nodes;
                        RTAR_combined_constraints = con_b_non_failed_nodes;
                    end
    %                 %Constraint c
    %                 if (current_dataObj.con_c_enabled_aux)
    %                     con_c_failed_nodes = current_dataObj.con_c_aux * RTAR_x <= current_dataObj.con_c_bounds_aux;
    %                     con_c_failed_nodes = sum(con_c_failed_nodes == 0, 2) > 0;
    %                     con_c_non_failed_nodes = ~con_c_failed_nodes;
    %                     RTAR_combined_constraints = and(RTAR_combined_constraints, con_c_non_failed_nodes);
    %                 end
                    %Constraint e
                    if (current_dataObj.con_e_enabled_aux)
                        con_e_result = current_dataObj.con_e_aux * RTAR_x <= current_dataObj.con_e_bounds_aux;
                        con_e_n_m_reshaped = reshape(con_e_result, [current_m, current_n])';
                        con_e_failed_nodes = sum(con_e_n_m_reshaped == 0, 2) > 0;
                        con_e_non_failed_nodes = ~con_e_failed_nodes;
                        RTAR_combined_constraints = and(con_e_non_failed_nodes, RTAR_combined_constraints);                   
                    end
                    %Do logical anding between all the constraints
                    %PBTA_combined_constraints = and(con_b_non_failed_nodes, con_c_non_failed_nodes);
                    %PBTA_combined_constraints = and(PBTA_combined_constraints, con_e_non_failed_nodes);

                    %RTAR H constraints
                    %Constraint b
                    if (current_dataObj.con_b_enabled_aux)
                        con_b_result = current_dataObj.con_b_aux * RTAR_H_x <= current_dataObj.con_b_bounds_aux;
                        con_b_n_m_reshaped = reshape(con_b_result, [current_m, current_n])';
                        con_b_failed_nodes = sum(con_b_n_m_reshaped == 0, 2) > 0;
                        con_b_non_failed_nodes = ~con_b_failed_nodes;
                        RTAR_H_combined_constraints = con_b_non_failed_nodes;
                    end
    %                 %Constraint c
    %                 if (current_dataObj.con_c_enabled_aux)
    %                     con_c_failed_nodes = current_dataObj.con_c_aux * RTAR_H_x <= current_dataObj.con_c_bounds_aux;
    %                     con_c_failed_nodes = sum(con_c_failed_nodes == 0, 2) > 0;
    %                     con_c_non_failed_nodes = ~con_c_failed_nodes;
    %                     RTAR_H_combined_constraints = and(RTAR_combined_constraints, con_c_non_failed_nodes);
    %                 end
                    %Constraint e
                    if (current_dataObj.con_e_enabled_aux)
                        con_e_result = current_dataObj.con_e_aux * RTAR_H_x <= current_dataObj.con_e_bounds_aux;
                        con_e_n_m_reshaped = reshape(con_e_result, [current_m, current_n])';
                        con_e_failed_nodes = sum(con_e_n_m_reshaped == 0, 2) > 0;
                        con_e_non_failed_nodes = ~con_e_failed_nodes;
                        RTAR_H_combined_constraints = and(con_e_non_failed_nodes, RTAR_combined_constraints);                   
                    end
                    %Do logical anding between all the constraints
                    %MCMF_combined_constraints = and(MCMF_con_b_non_failed_nodes, MCMF_con_c_non_failed_nodes);
                    %MCMF_combined_constraints = and(MCMF_combined_constraints, MCMF_con_e_non_failed_nodes);

                    for i = 1:current_n
                        if (and(RTAR_combined_constraints(i), ~RTAR_workers_failure_status(i)) == 0)
                            RTAR_workers_history_current_vals(i) = RTAR_workers_history_current_vals(i) + 0;
                        else
                            RTAR_workers_history_current_vals(i) = RTAR_workers_history_current_vals(i) + 1;
                        end
                        if (and(RTAR_H_combined_constraints, ~RTAR_H_workers_failure_status(i)) == 0)
                            RTAR_H_workers_history_current_vals(i) = RTAR_H_workers_history_current_vals(i) + 0;
                        else
                            RTAR_H_workers_history_current_vals(i) = RTAR_H_workers_history_current_vals(i) + 1;
                        end
                    end

                    %update inferred workers cpu frequencies
                    for worker_idx=0:current_n - 1
                        for task_idx=1:current_m
                            idx = (worker_idx .* current_m) + task_idx;
                            if (RTAR_x(idx) == 1)
                                workers_avg_inferred_cpu(worker_idx + 1) = workers_avg_inferred_cpu(worker_idx + 1) + current_dataObj.tasks_comp_delays(idx);
                            end
                            if (RTAR_H_x(idx) == 1)
                                workers_avg_inferred_cpu(worker_idx + 1) = workers_avg_inferred_cpu(worker_idx + 1) + current_dataObj.tasks_comp_delays(idx);
                            end
                            workers_avg_inferred_cpu = workers_avg_inferred_cpu ./ 2;
                        end
                    end
                end%simulations loop

                %update inferred workers cpu frequencies
                workers_avg_inferred_cpu = workers_avg_inferred_cpu ./ number_of_simulations;

                %update workers' history
                RTAR_workers_history_avg = floor(RTAR_workers_history_current_vals / number_of_simulations);
                RTAR_H_workers_history_avg = floor(RTAR_H_workers_history_current_vals / number_of_simulations);
                for i = 1:current_dataObj.N
                    current_dataObj.workers_history{i}(end+1) = RTAR_workers_history_avg(i);

                    %current_MCMF_dataObj.workers_history{i}(end+1) = RTAR_H_workers_history_avg(i);
                end
                for i = 1:current_dataObj.N
                    current_dataObj.workers_history_alphas(i) = sum(current_dataObj.workers_history{i} > 0);
                    current_dataObj.workers_history_betas(i) = sum(current_dataObj.workers_history{i} == 0);

                    %current_MCMF_dataObj.workers_history_alphas(i) = sum(current_MCMF_dataObj.workers_history{i} > 0);
                    %current_MCMF_dataObj.workers_history_betas(i) = sum(current_MCMF_dataObj.workers_history{i} == 0);
                end
                current_dataObj.workers_inferred_cpu_frequencies = workers_avg_inferred_cpu;
                %current_MCMF_dataObj.workers_inferred_cpu_frequencies = workers_avg_inferred_cpu;
            end%epochs loop
        end%tasks' loop
    end%workers' loop
end%hazards loop

%Task drop rate graphs
rep_max_average_task_drop_rate = zeros(N, M, H);
rep_kw_average_task_drop_rate = zeros(N, M, H);
true_average_task_drop_rate = zeros(N, M, H);
RTAR_average_task_drop_rate = zeros(N, M, H);
RTAR_H_average_task_drop_rate = zeros(N, M, H);



%Total recruitment cost graphs
RepMax_average_total_costs = zeros(N, M, H, 'double');
rep_kw_average_total_costs = zeros(N, M, H, 'double');
TRUE_average_total_costs = zeros(N, M, H, 'double');
RTAR_average_total_costs = zeros(N, M, H, 'double');
RTAR_H_average_total_costs = zeros(N, M, H, 'double');


%Optimal value graphs
RTAR_percentages_average = zeros(N, M, H);
RTAR_H_percentages_average = zeros(N, M, H);

%Number of replicas
RepMax_average_num_replicas = zeros(N, M, H, 'double');
rep_kw_average_num_replicas = zeros(N, M, H, 'double');
true_average_num_replicas = zeros(N, M, H, 'double');
RTAR_average_num_replicas = zeros(N, M, H, 'double');
RTAR_H_average_num_replicas = zeros(N, M, H, 'double');

for h = 1:H
    for sim_count=1:number_of_simulations
        for n = 1:N
            for m=1:M
                try 
                    current_m = all_results{n, m, h}.RTAR_result{1, 1}.dataObj.M;
                    current_n = all_results{n, m, h}.RTAR_result{1, 1}.dataObj.N;
                    current_dataObj = all_results{n, m, h}.RTAR_result{1, 1}.dataObj;
                    %MCMF_current_dataObj = all_results{n, m, h}.RTAR_H{1, 1}.dataObj;


                    %% Step 1: prepare decision vars
                    %% RepMax
                    rep_max_x = all_results{n, m, h}.repMax_result{1, 1}.repmax.all_sims{sim_count}.x;
                    rep_max_x = rep_max_x(1:(length(rep_max_x) - current_m));

                    rep_max_x_reshaped = reshape(rep_max_x, [current_m, current_n]);
                    rep_max_replicas = sum(rep_max_x_reshaped, 2);

                    RepMax_average_num_replicas(n, m) = RepMax_average_num_replicas(n, m) + sum(rep_max_replicas);

                    RepMax_tasks_workers_costs = rep_max_x_reshaped .* current_dataObj.workers_fitness_costs;
                    RepMax_current_total_cost = sum(RepMax_tasks_workers_costs, 'all');

                    % RepKW
                    rep_kw_x = all_results{n, m, h}.repMax_result{1, 1}.rep_kw.all_sims{sim_count}.x;
                    rep_kw_x = rep_kw_x(1:(length(rep_kw_x) - current_m));

                    rep_kw_x_reshaped = reshape(rep_kw_x, [current_m, current_n]);
                    rep_kw_replicas = sum(rep_kw_x_reshaped, 2);

                    rep_kw_average_num_replicas(n, m) = rep_kw_average_num_replicas(n, m) + sum(rep_kw_replicas);

                    rep_kw_tasks_workers_costs = rep_kw_x_reshaped .* current_dataObj.workers_fitness_costs;
                    rep_kw_current_total_cost = sum(rep_kw_tasks_workers_costs, 'all');



                    %% True
                    if (enable_true_benchmark)
                        true_x = all_results{n, m, h}.true_result{1, 1}.all_sims{sim_count}.x;
                        true_x_reshaped = reshape(true_x, [current_m, current_n]);
                        true_replicas = sum(true_x_reshaped, 2);

                        true_average_num_replicas(n, m) = true_average_num_replicas(n, m) + sum(true_replicas);
                        TRUE_tasks_workers_costs = true_x_reshaped .* current_dataObj.workers_fitness_costs;
                        TRUE_current_total_cost = sum(TRUE_tasks_workers_costs, 'all');
                    end

                    %% RTAR
                    %RTAR
                    RTAR_x = all_results{n, m, h}.RTAR_result{1, 1}.all_sims{sim_count}.optimal_solution.x;
                    RTAR_x_reshaped = reshape(RTAR_x, [current_m, current_n]);
                    RTAR_replicas = sum(RTAR_x_reshaped, 2);

                    RTAR_average_num_replicas(n, m) = RTAR_average_num_replicas(n, m) + sum(RTAR_replicas);
                    RTAR_optimal_val = current_dataObj.objectiveFunction_matlab * RTAR_x;

                    RTAR_tasks_workers_costs = RTAR_x_reshaped .* current_dataObj.workers_fitness_costs;
                    RTAR_current_total_cost = sum(RTAR_tasks_workers_costs, 'all');


                    %RTAR_H
                    RTAR_H_x = all_results{n, m, h}.RTAR_result{1, 1}.all_sims{sim_count}.RTAR_H.X;
                    RTAR_H_x_reshaped = reshape(RTAR_H_x, [current_m, current_n]);
                    RTAR_H_replicas = sum(RTAR_H_x_reshaped, 2);

                    RTAR_H_average_num_replicas(n, m) = RTAR_H_average_num_replicas(n, m) + sum(RTAR_H_replicas);

                    RTAR_optimal_val = current_dataObj.objectiveFunction_matlab * RTAR_H_x;

                    RTAR_H_tasks_workers_costs = RTAR_H_x_reshaped .* current_dataObj.workers_fitness_costs;
                    RTAR_H_current_total_cost = sum(RTAR_H_tasks_workers_costs, 'all');

                    %% Step2: Calculate failure probability
                    failure_rel_probs = 1 - current_dataObj.workers_tasks_rel_prop;
                    failure_rel_probs = reshape(failure_rel_probs, [current_m, current_n]);
                    %workers_bad_rep_cdf = reshape(current_dataObj.workers_bad_rep_cdf, [1, current_n]);
                    workers_bad_rep_cdf = reshape(current_dataObj.workers_history_expected_vals, [1, current_n]);
                    failure_rep_probs = ones(current_m, current_n) .* (1 - workers_bad_rep_cdf);
                    joint_failure_probs = failure_rel_probs .* failure_rep_probs;
                    failed_nodes_global = joint_failure_probs > failure_percentage;



                    %% RepMax
                    rep_max_tasks_fail_prob = failed_nodes_global;
                    rep_max_nodes_failed = sum(rep_max_tasks_fail_prob, 2);
                    rep_max_actual_nodes = rep_max_replicas - rep_max_nodes_failed;
                    rep_max_actual_nodes(rep_max_actual_nodes < 0) = 0;
                    rep_max_status = rep_max_actual_nodes == 0;
                    rep_max_percentage = sum(rep_max_status) ./ current_m;

                    rep_max_average_task_drop_rate(n, m) = rep_max_average_task_drop_rate(n, m) + rep_max_percentage;

                    %% Rep-KW
                    rep_kw_tasks_fail_prob = failed_nodes_global;
                    rep_kw_nodes_failed = sum(rep_kw_tasks_fail_prob, 2);
                    rep_kw_actual_nodes = rep_kw_replicas - rep_kw_nodes_failed;
                    rep_kw_actual_nodes(rep_kw_actual_nodes < 0) = 0;
                    rep_kw_status = rep_kw_actual_nodes == 0;
                    rep_kw_percentage = sum(rep_kw_status) ./ current_m;

                    rep_kw_average_task_drop_rate(n, m) = rep_kw_average_task_drop_rate(n, m) + rep_kw_percentage;


                    %% TRUE
                    if (enable_true_benchmark)
                        TRUE_tasks_fail_prob = failed_nodes_global;
                        TRUE_nodes_failed = sum(TRUE_tasks_fail_prob, 2);
                        TRUE_actual_nodes = true_replicas - TRUE_nodes_failed;
                        TRUE_actual_nodes(TRUE_actual_nodes < 0) = 0;
                        TRUE_status = TRUE_actual_nodes == 0;
                        TRUE_percentage = sum(TRUE_status) ./ current_m;
                    end

                    if (enable_true_benchmark)
                        true_average_task_drop_rate(n, m) = true_average_task_drop_rate(n, m) + TRUE_percentage;
                        TRUE_average_total_costs(n, m) = TRUE_average_total_costs(n, m) + TRUE_current_total_cost;
                    end

                    %% RTAR
                    %%RTAR
                    RTAR_tasks_fail_prob = failed_nodes_global;
                    RTAR_nodes_failed = sum(RTAR_tasks_fail_prob, 2);
                    RTAR_actual_nodes = RTAR_replicas - RTAR_nodes_failed;
                    RTAR_actual_nodes(RTAR_actual_nodes < 0) = 0;
                    RTAR_status = RTAR_actual_nodes == 0;
                    RTAR_percentage = sum(RTAR_status) ./ current_m;


                    RTAR_average_task_drop_rate(n, m) = RTAR_average_task_drop_rate(n, m) + RTAR_percentage;


                    %%RTAR H
                    RTAR_H_tasks_fail_prob = failed_nodes_global;
                    RTAR_H_nodes_failed = sum(RTAR_H_tasks_fail_prob, 2);
                    RTAR_H_actual_nodes = RTAR_H_replicas - RTAR_H_nodes_failed;
                    RTAR_H_actual_nodes(RTAR_H_actual_nodes < 0) = 0;
                    RTAR_H_status = RTAR_H_actual_nodes == 0;
                    RTAR_H_percentage = sum(RTAR_H_status) ./ current_m;


                    RTAR_H_average_task_drop_rate(n, m) = RTAR_H_average_task_drop_rate(n, m) + RTAR_H_percentage;    




                    RTAR_average_total_costs(n, m) = RTAR_average_total_costs(n, m) + RTAR_current_total_cost;
                    RTAR_H_average_total_costs(n, m) = RTAR_H_average_total_costs(n, m) + RTAR_H_current_total_cost;
                    RepMax_average_total_costs(n, m) = RepMax_average_total_costs(n, m) + RepMax_current_total_cost;
                    rep_kw_average_total_costs(n, m) = rep_kw_average_total_costs(n, m) + rep_kw_current_total_cost;

                    RTAR_percentages_average(n, m) = RTAR_percentages_average(n, m) + RTAR_optimal_val;
                    RTAR_H_percentages_average(n, m) = RTAR_H_percentages_average(n, m) + RTAR_optimal_val;
                catch ex
                    disp('error occured!');
                    rethrow(ex)
                end%try catch block
            end%end tasks for loop
        end%workers for loop
    end%simulations for loop
    rep_max_average_task_drop_rate(:, :, h) = rep_max_average_task_drop_rate(:, :, h) ./ number_of_simulations;
    RTAR_average_task_drop_rate(:, :, h) = RTAR_average_task_drop_rate(:, :, h) ./ number_of_simulations;
    RTAR_H_average_task_drop_rate(:, :, h) = RTAR_H_average_task_drop_rate(:, :, h) ./ number_of_simulations;
    
    RepMax_average_total_costs(:, :, h) = RepMax_average_total_costs(:, :, h) ./ number_of_simulations;
    RTAR_average_total_costs(:, :, h) = RTAR_average_total_costs(:, :, h) ./ number_of_simulations;
    RTAR_H_average_total_costs(:, :, h) = RTAR_H_average_total_costs(:, :, h) ./ number_of_simulations;
    
    RepMax_average_num_replicas(:, :, h) = RepMax_average_num_replicas(:, :, h) ./ number_of_simulations;
    RTAR_average_num_replicas(:, :, h) = RTAR_average_num_replicas(:, :, h) ./ number_of_simulations;
    RTAR_H_average_num_replicas(:, :, h) = RTAR_H_average_num_replicas(:, :, h) ./ number_of_simulations;
end%hazards loop
%calculate the averages

%sort columns
sort_averages = true;
if (sort_averages)
    
        
    if (N >= M)
        rep_max_average_task_drop_rate = sort(rep_max_average_task_drop_rate, 1);
        RepMax_average_total_costs = sort(RepMax_average_total_costs, 1);
        %rep_kw_average_task_drop_rate = sort(rep_kw_average_task_drop_rate, 1);
        if (enable_true_benchmark)
            true_average_task_drop_rate = sort(true_average_task_drop_rate, 1);
        end
        RTAR_average_task_drop_rate = sort(RTAR_average_task_drop_rate, 1);
        RTAR_average_total_costs = sort(RTAR_average_total_costs, 1);
        RTAR_H_average_task_drop_rate = sort(RTAR_H_average_task_drop_rate, 1);
        RTAR_H_average_total_costs = sort(RTAR_H_average_total_costs, 1);
    elseif (M > N)
        rep_max_average_task_drop_rate = sort(rep_max_average_task_drop_rate, 2);
        RepMax_average_total_costs = sort(RepMax_average_total_costs, 2);
        %rep_kw_average_task_drop_rate = sort(rep_kw_average_task_drop_rate, 2);
        if (enable_true_benchmark)
            true_average_task_drop_rate = sort(true_average_task_drop_rate, 2);
        end
        RTAR_average_task_drop_rate = sort(RTAR_average_task_drop_rate, 2);
        RTAR_average_total_costs = sort(RTAR_average_total_costs, 2);
        RTAR_H_average_task_drop_rate = sort(RTAR_H_average_task_drop_rate, 2);
        RTAR_H_average_total_costs = sort(RTAR_H_average_total_costs, 2);
    end    
end

RTAR_plot = '';
RTAR_H_plot = '';

repMax_plot = '';
rep_kw_plot = '';
true_plot = '';
repMax3D_plot1 = struct();
rtar3D_plot1 = struct();
rtarH3D_plot1 = struct();
repMax3D_plot1.x = zeros(H, M);
repMax3D_plot1.y = zeros(H, M);
repMax3D_plot1.z = zeros(H, M);

rtar3D_plot1.x = zeros(H, M);
rtar3D_plot1.y = zeros(H, M);
rtar3D_plot1.z = zeros(H, M);

rtarH3D_plot1.x = zeros(H, M);
rtarH3D_plot1.y = zeros(H, M);
rtarH3D_plot1.z = zeros(H, M);

for h=1:H
    for m = 1:M
        current_m = m_vector(m);
        repMax3D_plot1.x(h, m) = current_m ./ N_max;
        repMax3D_plot1.y(h, m) = hazard_rates(h);
        repMax3D_plot1.z(h, m) = rep_max_average_task_drop_rate(1, m, h);

        rtar3D_plot1.x(h, m) = current_m ./ N_max;
        rtar3D_plot1.y(h, m) = hazard_rates(h);
        rtar3D_plot1.z(h, m) = RTAR_average_task_drop_rate(1, m, h);

        rtarH3D_plot1.x(h, m) = current_m ./ N_max;
        rtarH3D_plot1.y(h, m) = hazard_rates(h);
        rtarH3D_plot1.z(h, m) = RTAR_H_average_task_drop_rate(1, m, h);


    end
end

[xz,y,z] = peaks;
xz = repMax3D_plot1.x;
y = repMax3D_plot1.y;
z = repMax3D_plot1.z;
f = figure;
[~, hc]     =   contourf(xz, y, z);
a1          =   gca;
a2          =   axes('Parent', f, 'Position', a1.Position);

hs          =   surf(xz, y, z, 'Parent', a2);

a1.Color    =   'none';
a2.Color    =   'none';

a1.ZLim     =   [0 1];
a2.ZLim     =   [-9 9];

a1.XTick    =   [];
a1.YTick    =   [];
a1.ZTick    =   [];

a1.Box      =   'off';
a2.Box      =   'off';

% Call after setting desired view on a2 (surf plot)
a1.View     =   a2.View;