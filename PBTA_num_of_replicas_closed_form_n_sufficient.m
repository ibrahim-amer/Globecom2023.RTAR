function [num_of_replicas_all_casses, actual_allocated_total_replicas_all_casses] = PBTA_num_of_replicas_closed_form_n_sufficient(n, m, tasks_priorities)
% This function returns the number of replicas allocated for each task 
% based on their priorities. 
% This function returns the number of replicas 'num_of_replicas' based on
% the tasks' priorities 'tasks_priorities' for 'm' number of tasks.
% The function will use a closed form (symbolic) equation to split the 'm'
% tasks on the 'n' workers based on the tasks' priorities
% 'tasks_priorities'.

% First, we calculate the value of the Lagrangian multiplier \lambda^{(1)}


% lambda_1 = (2 .* (n ./ m)) .* (-1 + ((1 ./ m) .* sum(tasks_priorities)));
% get_num_of_replicas_per_task = @(n, m, task_priority, lambda_1) ((n ./ m) .* task_priority) - ( (lambda_1) ./ 2);
get_num_of_replicas_per_task = @(n, m, task_priority, lambda_1, lambda_2) ((n ./ m) .* task_priority) + ((lambda_2 - lambda_1) ./ 2);

lambda_1 = [];
lambda_2 = [];

%% Case 1: lambda_1 = lambda_2 = 0
lambda_1 = 0;
lambda_2 = 0;

num_of_replicas_case_1 = zeros(1, m);
for j = 1: m
    r = floor(get_num_of_replicas_per_task(n, m, tasks_priorities(j), lambda_1, lambda_2));
    num_of_replicas_case_1(1, j) = (r > 0) .* r;
end

actual_allocated_total_replicas_case_1 = sum(num_of_replicas_case_1);

%% Case 2: lambda_1 = 0, lambda_2 \neq 0
lambda_1 = 0;
lambda_2 = (-1.5 .* (n ./ (m .* m))) .* sum(tasks_priorities);



num_of_replicas_case_2 = zeros(1, m);
for j = 1: m
    r = floor(get_num_of_replicas_per_task(n, m, tasks_priorities(j), lambda_1, lambda_2));
    num_of_replicas_case_2(1, j) = (r > 0) .* r;
end

actual_allocated_total_replicas_case_2 = sum(num_of_replicas_case_2);

%% Case 3: lambda_1 \neq 0, lambda_2 = 0
lambda_1 = (2 .* (n ./ m)) .* (((1 ./ m) .* sum(tasks_priorities)) - 1);
lambda_2 = 0;



num_of_replicas_case_3 = zeros(1, m);
for j = 1: m
    r = floor(get_num_of_replicas_per_task(n, m, tasks_priorities(j), lambda_1, lambda_2));
    num_of_replicas_case_3(1, j) = (r > 0) .* r;
end

actual_allocated_total_replicas_case_3 = sum(num_of_replicas_case_3);

%% Aggregate results

num_of_replicas_all_casses = cell(1, 3);
num_of_replicas_all_casses{1} = num_of_replicas_case_1;
num_of_replicas_all_casses{2} = num_of_replicas_case_2;
num_of_replicas_all_casses{3} = num_of_replicas_case_3;

actual_allocated_total_replicas_all_casses = zeros(1, 3);
actual_allocated_total_replicas_all_casses(1) = actual_allocated_total_replicas_case_1;
actual_allocated_total_replicas_all_casses(2) = actual_allocated_total_replicas_case_2;
actual_allocated_total_replicas_all_casses(3) = actual_allocated_total_replicas_case_3;
end

