
function [allResults] = RTAR_simulation(N_min, N_max, N_stepSize, M_min, M_max, M_stepSize, number_of_simulations, dataObj, checkConstraints)
%%
arguments
    N_min (1, 1) double = 2
    N_max (1, 1) double = 100
    N_stepSize (1, 1) double = 2
    M_min (1, 1) double = 2
    M_max (1, 1) double = 20
    M_stepSize (1, 1) = 2
    number_of_simulations (1, 1) double = 1.0
    dataObj = struct()
    checkConstraints (1, 1) logical = 1
end
    save_to_file = false;
    %%Preamble
    guid = string(java.util.UUID.randomUUID.toString);
    signature = strcat('[scenario1][N_min =  ', int2str(N_min), ', ', ...
        ' N_max =  ', int2str(N_max), ', ', ...
        ' N_stepSize =  ', int2str(N_stepSize), ', ', ...
        ' M_min =  ', int2str(M_min), ', ', ...
        ' M_max =  ', int2str(M_max),  ', ', ...
        ' M_stepSize =  ', int2str(M_stepSize),  ', ', ...
        ' n_sims =  ', int2str(number_of_simulations),  ']');
    guid = strcat(signature, '_', guid);
    %allResults_size = (ceil((N_max - N_min + 1) ./ N_stepSize) .* ceil((M_max - M_min + 1) ./ M_stepSize)) + 1;
    allResults = cell(ceil((N_max - N_min + 1) ./ N_stepSize), ceil((M_max - M_min + 1) ./ M_stepSize));
    ctr = 1;
    %%Preparing data
    dataObj.N = N_max;
    dataObj.M = M_max;
    dataObj = RTAR_prepare_data(dataObj);
    allow_debug = true;
    if (~isfield(dataObj, 'run_MCMF'))
        dataObj.run_MCMF = true;
    end
    %%Run simulations
    workers_ind = 1;
    
    for N = N_min:N_stepSize:N_max
        tasks_ind = 1;
        for M = M_min:M_stepSize:M_max
            fprintf('###########################################################\n');
            fprintf(strcat('N = ', int2str(N), ' ', ' OUT OF N = ', int2str(N_max), ' M = ', int2str(M), ...
                    ' ', ' OUT OF M = ', int2str(M_max), ' STARTS!!!', '\n'));
            simulation = struct();
            simulation.all_sims = cell(1, number_of_simulations);
            simulation.stats = struct();
            temp_dataObj = RTAR_slice_data(N, M, dataObj);
                       
           
            simulation.stats.averageOptimalVal = double(0);
            simulation.stats.averageRuntime = double(0);
            
            
            for sim = 1:number_of_simulations
                simulation.all_sims{sim}.optimal_solution = RTAR_ILP_solution_Gurobi(temp_dataObj, checkConstraints, allow_debug);                disp('RepMax_Rep_KW_scheme finished!');
                simulation.stats.total_number_of_allocated_workers = simulation.all_sims{sim}.optimal_solution.stats.total_number_of_allocated_workers;
                if (strcmp(simulation.all_sims{sim}.optimal_solution.status, 'OPTIMAL') || strcmp(simulation.all_sims{sim}.optimal_solution.status, 'INTERRUPTED'))
                    simulation.stats.averageOptimalVal = simulation.stats.averageOptimalVal + simulation.all_sims{sim}.optimal_solution.optimalVal;
                    simulation.stats.averageRuntime = simulation.stats.averageRuntime + simulation.all_sims{sim}.optimal_solution.runtime;
                else
                    simulation.stats.averageOptimalVal = simulation.stats.averageOptimalVal + 0;
                    simulation.stats.averageRuntime = simulation.stats.averageRuntime + 0;
                end
                 if (dataObj.run_heuristic)
                    simulation.all_sims{sim}.RTAR_H = struct();
                    simulation.all_sims{sim}.RTAR_H.X = heuristic_py(temp_dataObj);
                    simulation.all_sims{sim}.RTAR_H.X = double(simulation.all_sims{sim}.RTAR_H.X)';
                 end
            end
            if (strcmp(simulation.all_sims{sim}.optimal_solution.status, 'OPTIMAL') || strcmp(simulation.all_sims{sim}.optimal_solution.status, 'INTERRUPTED'))
                simulation.stats.averageOptimalVal = simulation.stats.averageOptimalVal ./ number_of_simulations;
                simulation.stats.averageRuntime = simulation.stats.averageRuntime ./ number_of_simulations;
            end
           
            simulation.dataObj = temp_dataObj;
            
            
            
            
            allResults{workers_ind, tasks_ind} = simulation;
            tasks_ind = tasks_ind + 1;
            ctr = ctr + 1;
            if (save_to_file)
                save(strcat(guid, '.mat'));
            end
            %clc;
            fprintf(strcat('N = ', int2str(N), ' ', ' OUT OF N = ', int2str(N_max), ...
                ' M = ', int2str(M), ' ', ' OUT OF K = ', int2str(M_max), ' ENDS!!!', '\n'));
            fprintf('###########################################################\n');
        end
        workers_ind = workers_ind + 1;
    end
    if (save_to_file)
        save(strcat(guid, '.mat'));
    end
    
end

