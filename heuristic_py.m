function [result] = heuristic_py(dataObj)
dataObj.rel_prop_t = [];
dataObj.worker_fitness_fn = [];
%Clear all anonymous functions before calling python function
dataObj.rel_prop_t = [];
dataObj.worker_fitness_fn = [];
dataObj.worker_fitness_fn_matlab = [];
dataObj.worker_fitness_fn_without_cost = [];
dataObj.obj_func_anonymous_func = [];
dataObj.G_func_tasks_p_level = [];
dataObj.B_func_p_budget = [];
result = py.RTAR_H.run_RTAR_H_vectorized(dataObj.tasks_budgets, dataObj.workers_fitness_costs, dataObj.workers_history_expected_vals);
end

