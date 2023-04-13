function [num_of_replicas_per_task total_allocated_replicas] = PBTA_optimal_num_of_replicas_closed_form(n, m, tasks_priorities)
f_optimal_num_of_replicas = [];
if (n >= m) 
    f_optimal_num_of_replicas = @(n, m, tasks_priorities) PBTA_num_of_replicas_closed_form_n_sufficient(n, m, tasks_priorities);
else 
    f_optimal_num_of_replicas = @(n, m, tasks_priorities) PBTA_num_of_replicas_closed_form_n_sufficient(n, m, tasks_priorities);
end
[num_of_replicas_all, total_allocated_replicas_all] = f_optimal_num_of_replicas(n, m, tasks_priorities);
% Filter solutions that have total_allocated_repliacs > n
%filtered_solutions contains index of filtered solutions
filter = zeros(1, 3);
for i = 1 : 3
    if (total_allocated_replicas_all(i) <= n)
        filter(i) = 1;
    end
end
filtered_solutions = total_allocated_replicas_all .* filter;
%Get the index of the solution with the maximum total_allocated_replicas
[m, idx] = max(filtered_solutions);
num_of_replicas_per_task = num_of_replicas_all(idx);
total_allocated_replicas = total_allocated_replicas_all(idx);
end

