function [result] = PBTA_test_num_of_replicas_closed_form(n_min, n_max, m_min, m_max)
N = n_max - n_min + 1;
M = m_max - m_min + 1;
Ns = n_min:n_max;
Ms = m_min:m_max;
result = repmat(struct('n', 0, 'm', 0, 'num_of_replicas', 0, 'tasks_priorities', [], 'actual_allocated_total_replicas', 0), N, M);

for i = 1 : N
   for j = 1 : M
       r = struct();
       r.n = Ns(i);
       r.m = Ms(j);
       r.tasks_priorities = randi([1 5], 1, Ms(j));
       %r.tasks_priorities = ones(1, Ms(j)) .* 4;
       [r.num_of_replicas r.actual_allocated_total_replicas] = PBTA_optimal_num_of_replicas_closed_form(Ns(i), Ms(j), r.tasks_priorities);
       result(i, j) = r;
   end
end
end

