%% This script renders a graph that compares PBTA, MCMF, RepMax, TRUE, PBTA-REPD and MCMF-REPD while varying the reputation of the workers and the number of tasks.
%Preparing the data
N_min = 50;
N_max = 100;
N_stepSize = 5;

M_min = 20;
M_max = 20;
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
 
all_results = cell(N, M);

for n=1:N
    for m=1:M
        
        all_results{n, m} = struct();
        workers_channel_states_from_val = 0.3;
        workers_channel_states_to_val = 0.8;
        
        current_dataObj = PBTA_init_dataObj_sim(n_vector(n), m_vector(m));
        current_dataObj.workers_inferred_cpu_frequencies = ones(1, n_vector(n)) .* 1e9;
        current_dataObj.workers_channel_states = workers_channel_states_from_val + (workers_channel_states_to_val - workers_channel_states_from_val) * rand(1, current_dataObj.N);  % size  = N
        current_MCMF_dataObj = current_dataObj;
        
        %RepMax
        all_results{n, m}.repMax_result = RepMax_simulation(n_vector(n), n_vector(n), N_stepSize, m_vector(m), ...
        m_vector(m), M_stepSize, number_of_simulations, current_dataObj, checkConstraints);

        %TRUE
        if (enable_true_benchmark)
            all_results{n, m}.true_result = TREED_battery_aware_simulation(n_vector(n), n_vector(n), N_stepSize, m_vector(m), m_vector(m),...
                M_stepSize, number_of_simulations, current_dataObj, checkConstraints);
        end


        
        for epoch=1:epochs
            %% ###################################################PBTA and MCMF: reputation only with no constraints ###################################################
            %Disabling constraints non-related to reputation
            current_dataObj.con_b_enabled = false;
            current_dataObj.con_c_enabled = false;
            current_dataObj.con_e_enabled = false;
            
            current_MCMF_dataObj.con_b_enabled = false;
            current_MCMF_dataObj.con_c_enabled = false;
            current_MCMF_dataObj.con_e_enabled = false;
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
            current_dataObj.worker_fitness_fn = @(rel_battery, worker_reputation, worker_cost, ... 
                    task_data_size, task_processing_density, worker_cpu, worker_snr) (exp(worker_reputation .^ 2) + worker_cpu + worker_snr) ;
            current_dataObj.worker_fitness_fn_matlab = @(rel_battery, worker_reputation, worker_cost, ... 
                task_data_size, task_processing_density, worker_cpu, worker_snr) (exp(worker_reputation .^ 2) + worker_cpu + worker_snr);
            current_MCMF_dataObj.worker_fitness_fn = @(rel_battery, worker_reputation, worker_cost, ... 
                    task_data_size, task_processing_density, worker_cpu, worker_snr) (exp(worker_reputation .^ 2) + worker_cpu + worker_snr);
            current_MCMF_dataObj.worker_fitness_fn_matlab = @(rel_battery, worker_reputation, worker_cost, ... 
                task_data_size, task_processing_density, worker_cpu, worker_snr) (exp(worker_reputation .^ 2) + worker_cpu + worker_snr);
            current_dataObj.run_MCMF = false;
            all_results{n, m}.PBTA_result = PBTA_simulation(n_vector(n), n_vector(n), N_stepSize, m_vector(m), ... 
                m_vector(m), M_stepSize, number_of_simulations, current_dataObj, checkConstraints);
            current_MCMF_dataObj.run_MCMF = true;
            current_MCMF_dataObj.MCMF_constraints_enabled = 0;
            all_results{n, m}.PBTA_MCMF_result = PBTA_simulation(n_vector(n), n_vector(n), N_stepSize, m_vector(m), ... 
                m_vector(m), M_stepSize, number_of_simulations, current_MCMF_dataObj, checkConstraints);

            %Check the status of the tasks after the recruitment by checking the
            %constraints
            current_m = all_results{n, m}.PBTA_result{1,1}.dataObj.M;
            current_n = all_results{n, m}.PBTA_result{1,1}.dataObj.N;
            current_dataObj = all_results{n, m}.PBTA_result{1, 1}.dataObj;
            current_MCMF_dataObj = all_results{n, m}.PBTA_MCMF_result{1, 1}.dataObj;
            PBTA_workers_history_current_vals = zeros(1, current_n);
            MCMF_workers_history_current_vals = zeros(1, current_n);
            workers_avg_inferred_cpu = ones(1, current_n) .* 1e9;
            for sim_count = 1:number_of_simulations
                %PBTA
                PBTA_x = all_results{n, m}.PBTA_result{1, 1}.all_sims{sim_count}.optimal_solution.x;
                PBTA_x_reshaped = reshape(PBTA_x, [current_m, current_n]);
                PBTA_x_n_m_reshaped = PBTA_x_reshaped';
                PBTA_x_num_of_tasks_per_worker = sum(PBTA_x_n_m_reshaped, 2);
                PBTA_replicas = sum(PBTA_x_reshaped, 2);


                %PBTA MCMF
                PBTA_MCMF_x = double(all_results{n, m}.PBTA_MCMF_result{1, 1}.all_sims{sim_count}.MCMF_result.X)';
                PBTA_MCMF_x_reshaped = reshape(PBTA_MCMF_x, [current_m, current_n]);
                PBTA_MCMF_x_n_m_reshaped = PBTA_MCMF_x_reshaped';
                PBTA_MCMF_x_num_of_tasks_per_worker = sum(PBTA_MCMF_x_n_m_reshaped, 2);
                PBTA_MCMF_replicas = sum(PBTA_MCMF_x_reshaped, 2);

                %Find failure probability
                failure_rel_probs = 1 - current_dataObj.workers_tasks_rel_prop;
                failure_rel_probs = reshape(failure_rel_probs, [current_m, current_n]);
                workers_bad_rep_cdf = reshape(current_dataObj.workers_history_expected_vals, [1, current_n]);
                failure_rep_probs = ones(current_m, current_n) .* (1 - workers_bad_rep_cdf);
                joint_failure_probs = failure_rel_probs .* failure_rep_probs;
                
                MCMF_failure_rel_probs = 1 - current_dataObj.workers_tasks_rel_prop;
                MCMF_failure_rel_probs = reshape(MCMF_failure_rel_probs, [current_m, current_n]);
                MCMF_workers_bad_rep_cdf = reshape(current_dataObj.workers_history_expected_vals, [1, current_n]);
                MCMF_failure_rep_probs = ones(current_m, current_n) .* (1 - MCMF_workers_bad_rep_cdf);
                MCMF_joint_failure_probs = failure_rel_probs .* MCMF_failure_rep_probs;
                
                PBTA_tasks_fail_prob = (PBTA_x_reshaped .* joint_failure_probs) > failure_percentage;
                PBTA_workers_failed = PBTA_tasks_fail_prob';
                PBTA_workers_failed = sum(PBTA_workers_failed, 2);
                PBTA_actual_workers = abs(PBTA_workers_failed - PBTA_x_num_of_tasks_per_worker);
                PBTA_workers_failure_status = PBTA_actual_workers == 0;
                PBTA_nodes_failed = sum(PBTA_tasks_fail_prob, 2);
                PBTA_actual_nodes = abs(PBTA_nodes_failed - PBTA_replicas);
                PBTA_status = PBTA_actual_nodes == 0;
                PBTA_percentage = sum(PBTA_status) ./ current_m;


                %%PBTA MCMF
                PBTA_MCMF_tasks_fail_prob = (PBTA_MCMF_x_reshaped .* MCMF_joint_failure_probs) > failure_percentage;
                PBTA_MCMF_workers_failed = PBTA_MCMF_tasks_fail_prob';
                PBTA_MCMF_workers_failed = sum(PBTA_MCMF_workers_failed, 2);
                PBTA_MCMF_actual_workers = abs(PBTA_MCMF_workers_failed - PBTA_MCMF_x_num_of_tasks_per_worker);
                PBTA_MCMF_workers_failure_status = PBTA_MCMF_actual_workers == 0;
                PBTA_MCMF_nodes_failed = sum(PBTA_MCMF_tasks_fail_prob, 2);
                PBTA_MCMF_actual_nodes = abs(PBTA_MCMF_nodes_failed - PBTA_MCMF_replicas);
                PBTA_MCMF_status = PBTA_MCMF_actual_nodes == 0;
                PBTA_MCMF_percentage = sum(PBTA_MCMF_status) ./ current_m;
                
                
                
                %Check constraints
                current_dataObj.con_b_enabled = true;
                current_dataObj.con_c_enabled = false;
                current_dataObj.con_e_enabled = false;
                
                current_MCMF_dataObj.con_b_enabled = true;
                current_MCMF_dataObj.con_c_enabled = false;
                current_MCMF_dataObj.con_e_enabled = false;
                
                PBTA_combined_constraints = [];
                MCMF_combined_constraints = [];
                   
                dataObj_temp = PBTA_prepare_data(current_dataObj);%re-prepare the data with constraints
                MCMF_dataObj_temp = PBTA_prepare_data(current_MCMF_dataObj);%re-prepare the data with constraints
                %Constraint b
                if (current_dataObj.con_b_enabled)
                    con_b_result = dataObj_temp.con_b * PBTA_x <= dataObj_temp.con_b_bounds;
                    con_b_n_m_reshaped = reshape(con_b_result, [current_m, current_n])';
                    con_b_failed_nodes = sum(con_b_n_m_reshaped == 0, 2) > 0;
                    con_b_non_failed_nodes = ~con_b_failed_nodes;
                    PBTA_combined_constraints = con_b_non_failed_nodes;
                end
                %Constraint c
                if (current_dataObj.con_c_enabled)
                    con_c_failed_nodes = dataObj_temp.con_c * PBTA_x <= dataObj_temp.con_c_bounds;
                    con_c_failed_nodes = sum(con_c_failed_nodes == 0, 2) > 0;
                    con_c_non_failed_nodes = ~con_c_failed_nodes;
                    PBTA_combined_constraints = and(PBTA_combined_constraints, con_c_non_failed_nodes);
                end
                %Constraint e
                if (current_dataObj.con_e_enabled)
                    con_e_result = dataObj_temp.con_e * PBTA_x <= dataObj_temp.con_e_bounds;
                    con_e_n_m_reshaped = reshape(con_e_result, [current_m, current_n])';
                    con_e_failed_nodes = sum(con_e_n_m_reshaped == 0, 2) > 0;
                    con_e_non_failed_nodes = ~con_e_failed_nodes;
                    PBTA_combined_constraints = and(con_e_non_failed_nodes, PBTA_combined_constraints);                   
                end
                %Do logical anding between all the constraints
                %PBTA_combined_constraints = and(con_b_non_failed_nodes, con_c_non_failed_nodes);
                %PBTA_combined_constraints = and(PBTA_combined_constraints, con_e_non_failed_nodes);
                
                %MCMF constraints
                %Constraint b
                if (current_MCMF_dataObj.con_b_enabled)
                    MCMF_con_b_result = dataObj_temp.con_b * PBTA_MCMF_x <= dataObj_temp.con_b_bounds;
                    MCMF_con_b_n_m_reshaped = reshape(MCMF_con_b_result, [current_m, current_n])';
                    MCMF_con_b_failed_nodes = sum(MCMF_con_b_n_m_reshaped == 0, 2) > 0;
                    MCMF_con_b_non_failed_nodes = ~MCMF_con_b_failed_nodes;
                    MCMF_combined_constraints = MCMF_con_b_non_failed_nodes;
                end
                %Constraint c
                if (current_MCMF_dataObj.con_c_enabled)
                    MCMF_con_c_failed_nodes = dataObj_temp.con_c * PBTA_MCMF_x <= dataObj_temp.con_c_bounds;
                    MCMF_con_c_failed_nodes = sum(MCMF_con_c_failed_nodes == 0, 2) > 0;
                    MCMF_con_c_non_failed_nodes = ~MCMF_con_c_failed_nodes;
                    MCMF_combined_constraints = and(MCMF_combined_constraints, MCMF_con_c_non_failed_nodes);
                end
                %Constraint e
                if (current_MCMF_dataObj.con_e_enabled)
                    MCMF_con_e_result = dataObj_temp.con_e * PBTA_MCMF_x <= dataObj_temp.con_e_bounds;
                    MCMF_con_e_n_m_reshaped = reshape(MCMF_con_e_result, [current_m, current_n])';
                    MCMF_con_e_failed_nodes = sum(MCMF_con_e_n_m_reshaped == 0, 2) > 0;
                    MCMF_con_e_non_failed_nodes = ~MCMF_con_e_failed_nodes;
                    MCMF_combined_constraints = and(MCMF_combined_constraints, MCMF_con_e_non_failed_nodes);
                end
                %Do logical anding between all the constraints
                %MCMF_combined_constraints = and(MCMF_con_b_non_failed_nodes, MCMF_con_c_non_failed_nodes);
                %MCMF_combined_constraints = and(MCMF_combined_constraints, MCMF_con_e_non_failed_nodes);
                
                for i = 1:current_n
                    if (and(PBTA_combined_constraints(i), ~PBTA_workers_failure_status(i)) == 0)
                        PBTA_workers_history_current_vals(i) = PBTA_workers_history_current_vals(i) + 0;
                    else
                        PBTA_workers_history_current_vals(i) = PBTA_workers_history_current_vals(i) + 1;
                    end
                    if (and(MCMF_combined_constraints, ~PBTA_MCMF_workers_failure_status(i)) == 0)
                        MCMF_workers_history_current_vals(i) = MCMF_workers_history_current_vals(i) + 0;
                    else
                        MCMF_workers_history_current_vals(i) = MCMF_workers_history_current_vals(i) + 1;
                    end
                end
                
                %update inferred workers cpu frequencies
                for worker_idx=0:current_n - 1
                    for task_idx=1:current_m
                        idx = (worker_idx .* current_m) + task_idx;
                        if (PBTA_x(idx) == 1)
                            workers_avg_inferred_cpu(worker_idx + 1) = workers_avg_inferred_cpu(worker_idx + 1) + current_dataObj.tasks_comp_delays(idx);
                        end
                        if (PBTA_MCMF_x(idx) == 1)
                            workers_avg_inferred_cpu(worker_idx + 1) = workers_avg_inferred_cpu(worker_idx + 1) + current_dataObj.tasks_comp_delays(idx);
                        end
                        workers_avg_inferred_cpu = workers_avg_inferred_cpu ./ 2;
                    end
                end
            end%simulations loop
            
            %update inferred workers cpu frequencies
            workers_avg_inferred_cpu = workers_avg_inferred_cpu ./ number_of_simulations;
            
            %update workers' history
            PBTA_workers_history_avg = floor(PBTA_workers_history_current_vals / number_of_simulations);
            MCMF_workers_history_avg = floor(MCMF_workers_history_current_vals / number_of_simulations);
            for i = 1:current_dataObj.N
                current_dataObj.workers_history{i}(end+1) = PBTA_workers_history_avg(i);

                current_MCMF_dataObj.workers_history{i}(end+1) = MCMF_workers_history_avg(i);
            end
            for i = 1:current_dataObj.N
                current_dataObj.workers_history_alphas(i) = sum(current_dataObj.workers_history{i} > 0);
                current_dataObj.workers_history_betas(i) = sum(current_dataObj.workers_history{i} == 0);

                current_MCMF_dataObj.workers_history_alphas(i) = sum(current_MCMF_dataObj.workers_history{i} > 0);
                current_MCMF_dataObj.workers_history_betas(i) = sum(current_MCMF_dataObj.workers_history{i} == 0);
            end
            current_dataObj.workers_inferred_cpu_frequencies = workers_avg_inferred_cpu;
            current_MCMF_dataObj.workers_inferred_cpu_frequencies = workers_avg_inferred_cpu;
        end%epochs loop
    end%tasks' loop
end%workers' loop

%Task drop rate graphs
rep_max_average_task_drop_rate = zeros(N, M);
rep_kw_average_task_drop_rate = zeros(N, M);
true_average_task_drop_rate = zeros(N, M);
PBTA_average_task_drop_rate = zeros(N, M);
PBTA_MCMF_average_task_drop_rate = zeros(N, M);



%Total recruitment cost graphs
RepMax_average_total_costs = zeros(N, M, 'double');
rep_kw_average_total_costs = zeros(N, M, 'double');
TRUE_average_total_costs = zeros(N, M, 'double');
PBTA_average_total_costs = zeros(N, M, 'double');
PBTA_MCMF_average_total_costs = zeros(N, M, 'double');


%Optimal value graphs
PBTA_percentages_average = zeros(N, M);
PBTA_MCMF_percentages_average = zeros(N, M);

%Number of replicas
RepMax_average_num_replicas = zeros(N, M, 'double');
rep_kw_average_num_replicas = zeros(N, M, 'double');
true_average_num_replicas = zeros(N, M, 'double');
PBTA_average_num_replicas = zeros(N, M, 'double');
PBTA_MCMF_average_num_replicas = zeros(N, M, 'double');


for sim_count=1:number_of_simulations
    for n = 1:N
        for m=1:M
            try 
                current_m = all_results{n, m}.PBTA_result{1, 1}.dataObj.M;
                current_n = all_results{n, m}.PBTA_result{1, 1}.dataObj.N;
                current_dataObj = all_results{n, m}.PBTA_result{1, 1}.dataObj;
                MCMF_current_dataObj = all_results{n, m}.PBTA_MCMF_result{1, 1}.dataObj;


                %% Step 1: prepare decision vars
                %% RepMax
                rep_max_x = all_results{n, m}.repMax_result{1, 1}.repmax.all_sims{sim_count}.x;
                rep_max_x = rep_max_x(1:(length(rep_max_x) - current_m));

                rep_max_x_reshaped = reshape(rep_max_x, [current_m, current_n]);
                rep_max_replicas = sum(rep_max_x_reshaped, 2);
                
                RepMax_average_num_replicas(n, m) = RepMax_average_num_replicas(n, m) + sum(rep_max_replicas);
                
                RepMax_tasks_workers_costs = rep_max_x_reshaped .* current_dataObj.workers_fitness_costs;
                RepMax_current_total_cost = sum(RepMax_tasks_workers_costs, 'all');
                
                %% RepKW
                rep_kw_x = all_results{n, m}.repMax_result{1, 1}.rep_kw.all_sims{sim_count}.x;
                rep_kw_x = rep_kw_x(1:(length(rep_kw_x) - current_m));

                rep_kw_x_reshaped = reshape(rep_kw_x, [current_m, current_n]);
                rep_kw_replicas = sum(rep_kw_x_reshaped, 2);
                
                rep_kw_average_num_replicas(n, m) = rep_kw_average_num_replicas(n, m) + sum(rep_kw_replicas);
                
                rep_kw_tasks_workers_costs = rep_kw_x_reshaped .* current_dataObj.workers_fitness_costs;
                rep_kw_current_total_cost = sum(rep_kw_tasks_workers_costs, 'all');
                


                %% True
                if (enable_true_benchmark)
                    true_x = all_results{n, m}.true_result{1, 1}.all_sims{sim_count}.x;
                    true_x_reshaped = reshape(true_x, [current_m, current_n]);
                    true_replicas = sum(true_x_reshaped, 2);
                    
                    true_average_num_replicas(n, m) = true_average_num_replicas(n, m) + sum(true_replicas);
                    TRUE_tasks_workers_costs = true_x_reshaped .* current_dataObj.workers_fitness_costs;
                    TRUE_current_total_cost = sum(TRUE_tasks_workers_costs, 'all');
                end

                %% PBTA
                %PBTA
                PBTA_x = all_results{n, m}.PBTA_result{1, 1}.all_sims{sim_count}.optimal_solution.x;
                PBTA_x_reshaped = reshape(PBTA_x, [current_m, current_n]);
                PBTA_replicas = sum(PBTA_x_reshaped, 2);
                
                PBTA_average_num_replicas(n, m) = PBTA_average_num_replicas(n, m) + sum(PBTA_replicas);
                pbta_optimal_val = current_dataObj.objectiveFunction_matlab * PBTA_x;
                
                PBTA_tasks_workers_costs = PBTA_x_reshaped .* current_dataObj.workers_fitness_costs;
                PBTA_current_total_cost = sum(PBTA_tasks_workers_costs, 'all');


                %PBTA MCMF
                PBTA_MCMF_x = double(all_results{n, m}.PBTA_MCMF_result{1, 1}.all_sims{sim_count}.MCMF_result.X)';
                PBTA_MCMF_x_reshaped = reshape(PBTA_MCMF_x, [current_m, current_n]);
                PBTA_MCMF_replicas = sum(PBTA_MCMF_x_reshaped, 2);
                
                PBTA_MCMF_average_num_replicas(n, m) = PBTA_MCMF_average_num_replicas(n, m) + sum(PBTA_MCMF_replicas);
                
                mcmf_optimal_val = current_dataObj.objectiveFunction_matlab * PBTA_MCMF_x;
                
                PBTA_MCMF_tasks_workers_costs = PBTA_MCMF_x_reshaped .* current_dataObj.workers_fitness_costs;
                PBTA_MCMF_current_total_cost = sum(PBTA_MCMF_tasks_workers_costs, 'all');

                %% Step2: Calculate failure probability
                failure_rel_probs = 1 - current_dataObj.workers_tasks_rel_prop;
                failure_rel_probs = reshape(failure_rel_probs, [current_m, current_n]);
                %workers_bad_rep_cdf = reshape(current_dataObj.workers_bad_rep_cdf, [1, current_n]);
                workers_bad_rep_cdf = reshape(current_dataObj.workers_history_expected_vals, [1, current_n]);
                failure_rep_probs = ones(current_m, current_n) .* (1 - workers_bad_rep_cdf);
                joint_failure_probs = failure_rel_probs .* failure_rep_probs;
                
                MCMF_failure_rel_probs = 1 - current_dataObj.workers_tasks_rel_prop;
                MCMF_failure_rel_probs = reshape(MCMF_failure_rel_probs, [current_m, current_n]);
                %workers_bad_rep_cdf = reshape(current_dataObj.workers_bad_rep_cdf, [1, current_n]);
                MCMF_workers_bad_rep_cdf = reshape(current_dataObj.workers_history_expected_vals, [1, current_n]);
                MCMF_failure_rep_probs = ones(current_m, current_n) .* (1 - MCMF_workers_bad_rep_cdf);
                MCMF_joint_failure_probs = MCMF_failure_rel_probs .* MCMF_failure_rep_probs;

                %% RepMax
                rep_max_tasks_fail_prob = (rep_max_x_reshaped .* joint_failure_probs) > failure_percentage;
                rep_max_nodes_failed = sum(rep_max_tasks_fail_prob, 2);
                rep_max_actual_nodes = abs(rep_max_nodes_failed - rep_max_replicas);
                rep_max_status = rep_max_actual_nodes == 0;
                rep_max_percentage = sum(rep_max_status) ./ current_m;

                rep_max_average_task_drop_rate(n, m) = rep_max_average_task_drop_rate(n, m) + rep_max_percentage;


                %% TRUE
                if (enable_true_benchmark)
                    TRUE_tasks_fail_prob = (true_x_reshaped .* joint_failure_probs) > failure_percentage;
                    TRUE_nodes_failed = sum(TRUE_tasks_fail_prob, 2);
                    TRUE_actual_nodes = abs(TRUE_nodes_failed - true_replicas);
                    TRUE_status = TRUE_actual_nodes == 0;
                    TRUE_percentage = sum(TRUE_status) ./ current_m;
                end

                if (enable_true_benchmark)
                    true_average_task_drop_rate(n, m) = true_average_task_drop_rate(n, m) + TRUE_percentage;
                    TRUE_average_total_costs(n, m) = TRUE_average_total_costs(n, m) + TRUE_current_total_cost;
                end

                %% PBTA
                %%PBTA
                PBTA_tasks_fail_prob = (PBTA_x_reshaped .* joint_failure_probs) > failure_percentage;
                PBTA_nodes_failed = sum(PBTA_tasks_fail_prob, 2);
                PBTA_actual_nodes = abs(PBTA_nodes_failed - PBTA_replicas);
                PBTA_status = PBTA_actual_nodes == 0;
                PBTA_percentage = sum(PBTA_status) ./ current_m;


                PBTA_average_task_drop_rate(n, m) = PBTA_average_task_drop_rate(n, m) + PBTA_percentage;

                %%PBTA MCMF
                PBTA_MCMF_tasks_fail_prob = (PBTA_MCMF_x_reshaped .* MCMF_joint_failure_probs) > failure_percentage;
                PBTA_MCMF_nodes_failed = sum(PBTA_MCMF_tasks_fail_prob, 2);
                PBTA_MCMF_actual_nodes = abs(PBTA_MCMF_nodes_failed - PBTA_MCMF_replicas);
                PBTA_MCMF_status = PBTA_MCMF_actual_nodes == 0;
                PBTA_MCMF_percentage = sum(PBTA_MCMF_status) ./ current_m;


                PBTA_MCMF_average_task_drop_rate(n, m) = PBTA_MCMF_average_task_drop_rate(n, m) + PBTA_MCMF_percentage;    
                
                
                PBTA_average_total_costs(n, m) = PBTA_average_total_costs(n, m) + PBTA_current_total_cost;
                PBTA_MCMF_average_total_costs(n, m) = PBTA_MCMF_average_total_costs(n, m) + PBTA_MCMF_current_total_cost;
                RepMax_average_total_costs(n, m) = RepMax_average_total_costs(n, m) + RepMax_current_total_cost;
                rep_kw_average_total_costs(n, m) = rep_kw_average_total_costs(n, m) + rep_kw_current_total_cost;
                
                PBTA_percentages_average(n, m) = PBTA_percentages_average(n, m) + pbta_optimal_val;
                PBTA_MCMF_percentages_average(n, m) = PBTA_MCMF_percentages_average(n, m) + mcmf_optimal_val;
            catch ex
                disp('error occured!');
                rethrow(ex)
            end%try catch block
        end%end tasks for loop
    end%workers for loop
end%simulations for loop
 
%calculate the averages
rep_max_average_task_drop_rate = rep_max_average_task_drop_rate ./ number_of_simulations;
if (enable_true_benchmark)
    true_average_task_drop_rate = true_average_task_drop_rate ./ number_of_simulations;
    TRUE_average_total_costs = TRUE_average_total_costs ./ number_of_simulations;
    true_average_num_replicas = true_average_num_replicas ./ number_of_simulations;
end
PBTA_average_task_drop_rate = PBTA_average_task_drop_rate ./ number_of_simulations;
PBTA_MCMF_average_task_drop_rate = PBTA_MCMF_average_task_drop_rate ./ number_of_simulations;


PBTA_average_total_costs = PBTA_average_total_costs ./ number_of_simulations;
PBTA_MCMF_average_total_costs = PBTA_MCMF_average_total_costs ./ number_of_simulations;
RepMax_average_total_costs = RepMax_average_total_costs ./ number_of_simulations;

RepMax_average_num_replicas = RepMax_average_num_replicas ./ number_of_simulations;
rep_kw_average_num_replicas = rep_kw_average_num_replicas ./ number_of_simulations;
PBTA_average_num_replicas = PBTA_average_num_replicas ./ number_of_simulations;
PBTA_MCMF_average_num_replicas = PBTA_MCMF_average_num_replicas ./ number_of_simulations;
%sort columns
sort_averages = true;
if (sort_averages)
    
        
    if (N >= M)
        rep_max_average_task_drop_rate = sort(rep_max_average_task_drop_rate, 1);
        rep_kw_average_task_drop_rate = sort(rep_kw_average_task_drop_rate, 1);
        if (enable_true_benchmark)
            true_average_task_drop_rate = sort(true_average_task_drop_rate, 1);
        end
        PBTA_average_task_drop_rate = sort(PBTA_average_task_drop_rate, 1);
        PBTA_MCMF_average_task_drop_rate = sort(PBTA_MCMF_average_task_drop_rate, 1);
    elseif (M > N)
        rep_max_average_task_drop_rate = sort(rep_max_average_task_drop_rate, 2);
        rep_kw_average_task_drop_rate = sort(rep_kw_average_task_drop_rate, 2);
        if (enable_true_benchmark)
            true_average_task_drop_rate = sort(true_average_task_drop_rate, 2);
        end
        PBTA_average_task_drop_rate = sort(PBTA_average_task_drop_rate, 2);
        PBTA_MCMF_average_task_drop_rate = sort(PBTA_MCMF_average_task_drop_rate, 2);
    end    
end

PBTA_plot = '';
PBTA_MCMF_plot = '';

repMax_plot = '';
rep_kw_plot = '';
true_plot = '';
if (N >= M)
    for n = 1:N
        current_n = n_vector(n);
        repMax_plot = strcat(repMax_plot, '(', num2str(current_n),', ', num2str(rep_max_average_task_drop_rate(n, 1)), ')');
        rep_kw_plot = strcat(rep_kw_plot, '(', num2str(current_n),', ', num2str(rep_kw_average_task_drop_rate(n, 1)), ')');
        if (enable_true_benchmark)
            true_plot = strcat(true_plot, '(', num2str(current_n),', ', num2str(true_average_task_drop_rate(n, 1)), ')');
        end
        PBTA_plot = strcat(PBTA_plot, '(', num2str(current_n),', ', num2str(PBTA_average_task_drop_rate(n, 1)), ')');
        PBTA_MCMF_plot = strcat(PBTA_MCMF_plot, '(', num2str(current_n),', ', num2str(PBTA_MCMF_average_task_drop_rate(n, 1)), ')');
    end
else
    for m = 1:M
        current_m = m_vector(n);
        repMax_plot = strcat(repMax_plot, '(', num2str(current_m),', ', num2str(rep_max_average_task_drop_rate(1, m)), ')');
        rep_kw_plot = strcat(rep_kw_plot, '(', num2str(current_m),', ', num2str(rep_kw_average_task_drop_rate(1, m)), ')');
        if (enable_true_benchmark)
            true_plot = strcat(true_plot, '(', num2str(current_m),', ', num2str(true_average_task_drop_rate(1, m)), ')');
        end
        PBTA_plot = strcat(PBTA_plot, '(', num2str(current_m),', ', num2str(PBTA_average_task_drop_rate(1, m)), ')');
        PBTA_MCMF_plot = strcat(PBTA_MCMF_plot, '(', num2str(current_m),', ', num2str(PBTA_MCMF_average_task_drop_rate(1, m)), ')');
    end
end%end if else
disp(strcat('###############Average task drop rate', '###############'));
disp('PBTA Plot: ');
disp(PBTA_plot);
disp('-------------------------------------------------------');
disp('PBTA MCMF Plot: ');
disp(PBTA_MCMF_plot);
disp('-------------------------------------------------------');

disp('RepMax Plot: ');
disp(repMax_plot);
disp('RepKW Plot: ');
disp(rep_kw_plot);
disp('-------------------------------------------------------');
if (enable_true_benchmark)
    disp('true_plot Plot: ');
    disp(true_plot);
end
disp('################################################');
disp(strcat('###############Total cost', '###############'));
PBTA_plot = '';
PBTA_MCMF_plot = '';

repMax_plot = '';
rep_kw_plot = '';
true_plot = '';
if (N >= M)
    for n = 1:N
        current_n = n_vector(n);
        repMax_plot = strcat(repMax_plot, '(', num2str(current_n),', ', num2str(RepMax_average_total_costs(n, 1)), ')');
        rep_kw_plot = strcat(rep_kw_plot, '(', num2str(current_n),', ', num2str(rep_kw_average_total_costs(n, 1)), ')');
        if (enable_true_benchmark)
            true_plot = strcat(true_plot, '(', num2str(current_n),', ', num2str(TRUE_average_total_costs(n, 1)), ')');
        end
        PBTA_plot = strcat(PBTA_plot, '(', num2str(current_n),', ', num2str(PBTA_average_total_costs(n, 1)), ')');
        PBTA_MCMF_plot = strcat(PBTA_MCMF_plot, '(', num2str(current_n),', ', num2str(PBTA_MCMF_average_total_costs(n, 1)), ')');
    end
else
    for m = 1:M
        current_m = m_vector(n);
        repMax_plot = strcat(repMax_plot, '(', num2str(current_m),', ', num2str(RepMax_average_total_costs(1, m)), ')');
        rep_kw_plot = strcat(rep_kw_plot, '(', num2str(current_m),', ', num2str(rep_kw_average_total_costs(1, m)), ')');
        if (enable_true_benchmark)
            true_plot = strcat(true_plot, '(', num2str(current_m),', ', num2str(TRUE_average_total_costs(1, m)), ')');
        end
        PBTA_plot = strcat(PBTA_plot, '(', num2str(current_m),', ', num2str(PBTA_average_total_costs(1, m)), ')');
        PBTA_MCMF_plot = strcat(PBTA_MCMF_plot, '(', num2str(current_m),', ', num2str(PBTA_MCMF_average_total_costs(1, m)), ')');
    end
end%end if else

disp('PBTA Plot: ');
disp(PBTA_plot);
disp('-------------------------------------------------------');
disp('PBTA MCMF Plot: ');
disp(PBTA_MCMF_plot);
disp('-------------------------------------------------------');

disp('RepMax Plot: ');
disp(repMax_plot);
disp('-------------------------------------------------------');
if (enable_true_benchmark)
    disp('true_plot Plot: ');
    disp(true_plot);
end

disp('################################################');
disp(strcat('###############Number of Replicas', '###############'));
PBTA_plot = '';
PBTA_MCMF_plot = '';

repMax_plot = '';
rep_kw_plot = '';
true_plot = '';
if (N >= M)
    for n = 1:N
        current_n = n_vector(n);
        repMax_plot = strcat(repMax_plot, '(', num2str(current_n),', ', num2str(RepMax_average_num_replicas(n, 1)), ')');
        rep_kw_plot = strcat(rep_kw_plot, '(', num2str(current_n),', ', num2str(rep_kw_average_num_replicas(n, 1)), ')');
        if (enable_true_benchmark)
            true_plot = strcat(true_plot, '(', num2str(current_n),', ', num2str(true_average_num_replicas(n, 1)), ')');
        end
        PBTA_plot = strcat(PBTA_plot, '(', num2str(current_n),', ', num2str(PBTA_average_num_replicas(n, 1)), ')');
        PBTA_MCMF_plot = strcat(PBTA_MCMF_plot, '(', num2str(current_n),', ', num2str(PBTA_MCMF_average_num_replicas(n, 1)), ')');
    end
else
    for m = 1:M
        current_m = m_vector(n);
        repMax_plot = strcat(repMax_plot, '(', num2str(current_m),', ', num2str(RepMax_average_num_replicas(1, m)), ')');
        rep_kw_plot = strcat(rep_kw_plot, '(', num2str(current_m),', ', num2str(rep_kw_average_num_replicas(1, m)), ')');
        if (enable_true_benchmark)
            true_plot = strcat(true_plot, '(', num2str(current_m),', ', num2str(true_average_num_replicas(1, m)), ')');
        end
        PBTA_plot = strcat(PBTA_plot, '(', num2str(current_m),', ', num2str(PBTA_average_num_replicas(1, m)), ')');
        PBTA_MCMF_plot = strcat(PBTA_MCMF_plot, '(', num2str(current_m),', ', num2str(PBTA_MCMF_average_num_replicas(1, m)), ')');
    end
end%end if else

disp('PBTA Plot: ');
disp(PBTA_plot);
disp('-------------------------------------------------------');
disp('PBTA MCMF Plot: ');
disp(PBTA_MCMF_plot);
disp('-------------------------------------------------------');

disp('RepMax Plot: ');
disp(repMax_plot);
disp('-------------------------------------------------------');
if (enable_true_benchmark)
    disp('true_plot Plot: ');
    disp(true_plot);
end


% disp('################################################');
% disp(strcat('###############PBTA - MCMF Optimal Value', '###############'));
% PBTA_plot = '';
% PBTA_MCMF_plot = '';
% 
% repMax_plot = '';
% true_plot = '';
% if (N >= M)
%     for n = 1:N
%         current_n = n_vector(n);
%         PBTA_plot = strcat(PBTA_plot, '(', num2str(current_n),', ', num2str(PBTA_percentages_average(n, 1)), ')');
%         PBTA_MCMF_plot = strcat(PBTA_MCMF_plot, '(', num2str(current_n),', ', num2str(PBTA_MCMF_percentages_average(n, 1)), ')');
%     end
% else
%     for m = 1:M
%         current_m = m_vector(m);
%         PBTA_plot = strcat(PBTA_plot, '(', num2str(current_m),', ', num2str(PBTA_percentages_average(1, m)), ')');
%         PBTA_MCMF_plot = strcat(PBTA_MCMF_plot, '(', num2str(current_m),', ', num2str(PBTA_MCMF_percentages_average(1, m)), ')');
%     end
% end%end if else
% 
% disp('PBTA Plot: ');
% disp(PBTA_plot);
% disp('-------------------------------------------------------');
% disp('PBTA MCMF Plot: ');
% disp(PBTA_MCMF_plot);
% disp('-------------------------------------------------------');