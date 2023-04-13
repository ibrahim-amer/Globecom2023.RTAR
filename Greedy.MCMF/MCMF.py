# import debugpy
# debugpy.debug_this_thread()
# debug matlab-python 
# https://www.mathworks.com/matlabcentral/answers/1645680-how-can-i-debug-python-code-using-matlab-s-python-interface-and-visual-studio-code#:~:text=Attach%20the%20MATLAB%20process%20to,you%20specified%20in%20the%20launch. 
# Awesome article about MCMF: https://cp-algorithms.com/graph/min_cost_flow.html
import Scratch as scratch
import DictToObj as DTO
import numpy as np
import queue
import timeit
# if you ever encountered the error: "Python Error: TclError: Can't find a usable init.tcl in the following directories:", 
# please refer to this solution: "https://stackoverflow.com/a/40824957/1304187"
import GraphVisualization as GV
'''
To debug this code from Matlab, follow the following steps:
1. Open matlab and run function reload_python_workspace
2. Open Visual Studio Code then go to debug tab on the left hand-side ribbon.
3. Press attach to process and choose the Matlab process.
4. Set the potential breakpoints in your python code.
5. Go back to Matlab and run the function that calls the Python function. 
'''


def bellmanFord_shortestPath(numberOfVertices, srcNodeIdx, tasksNodesStartIdx, 
                            tasksNodesEndIdx, workersNodesStartIdx, workersNodesEndIdx,
                            cap, cost, energy_cap = None,
                            cap_hat = None, MCMF_constraints_enabled = None, 
                            budget_con_enabled = None):
    distance = np.empty(numberOfVertices)
    inq = np.empty(numberOfVertices)
    inq.fill(bool(False))

    distance.fill(float('Inf'))
    predecessor = np.empty(numberOfVertices)

    distance[srcNodeIdx] = 0
    q = queue.Queue()
    q.put(srcNodeIdx)
    predecessor.fill(-1)

    while (not q.empty()):
        u = q.queue[0]
        q.get()
        inq[u] = False
        for v in range(numberOfVertices):
            if MCMF_constraints_enabled is not None and MCMF_constraints_enabled == 1:
                if cap[u][v] > 0 and distance[v] > distance[u] + cost[u][v] and energy_cap[u][v] > 0:
                        distance[v] = distance[u] + cost[u][v]
                        predecessor[v] = u
                        if (not inq[v]):
                            inq[v] = True
                            q.put(v)
            if budget_con_enabled is not None and budget_con_enabled == 1:
                if cap[u][v] > 0 and distance[v] > distance[u] + cost[u][v] and cap_hat[u][v] > 0:
                        if u >= tasksNodesStartIdx and u <= tasksNodesEndIdx and \
                           v >= workersNodesStartIdx and v <= workersNodesEndIdx and \
                           cap_hat[srcNodeIdx][u] < cap_hat[u][v]:#the budget isn't sufficient
                            continue
                        distance[v] = distance[u] + cost[u][v]
                        predecessor[v] = u
                        if (not inq[v]):
                            inq[v] = True
                            q.put(v)
            else:
                if cap[u][v] > 0 and distance[v] > distance[u] + cost[u][v]:
                        distance[v] = distance[u] + cost[u][v]
                        predecessor[v] = u
                        if (not inq[v]):
                            inq[v] = True
                            q.put(v)
    #distance = distance.astype('int')
    predecessor = predecessor.astype('int')
    return distance, predecessor




def bellmanFord(source, weights):
    '''
    https://stackoverflow.com/a/40042174/1304187
    This implementation takes in a graph and fills two arrays
    (distance and predecessor) with shortest-path (less cost/distance/metric) information

    https://en.wikipedia.org/wiki/Bellman%E2%80%93Ford_algorithm
    '''
    n = weights.shape[0]

    # Step 1: initialize graph
    distance = np.empty(n)
    distance.fill(float('Inf'))      # At the beginning, all vertices have a weight of infinity
    predecessor = np.empty(n)
    predecessor.fill(float('NaN'))   # And a null predecessor

    distance[source] = 0             # Except for the Source, where the Weight is zero

    # Step 2: relax edges repeatedly
    for _ in range(1, n):
        for (u, v), w in np.ndenumerate(weights):
            if distance[u] + w < distance[v]:
                distance[v] = distance[u] + w
    predecessor[v] = u

    # Step 3: check for negative-weight cycles
    for (u, v), w in np.ndenumerate(weights):
        if distance[u] + w < distance[v]:
             ValueError("Graph contains a negative-weight cycle")

    return distance, predecessor

def init_sim(dataObj):
    dataObj.tasks_budgets = np.array(dataObj.tasks_budgets)
    dataObj.workers_fitness_costs = np.array(dataObj.workers_fitness_costs)
    dataObj.tasks_priorities = np.array(dataObj.tasks_priorities)
    return dataObj
def run_MCMF(dataObjDict):
    dataObj = DTO.DictToObj(dataObjDict)
    dataObj.N = int(dataObj.N)
    dataObj.M = int(dataObj.M)
    dataObj = init_sim(dataObj)
    if dataObj.MCMF_constraints_enabled == 1:
        dataObj = MCMF_preprocessing(dataObj)
    dataObj.updated_max_energies = dataObj.max_energies
    numOfVertices = dataObj.N + dataObj.M + 2
    srcNodeIdx = 0
    sinkNodeIdx = numOfVertices - 1

    tasksStartIdx = 1
    tasksEndIdx = dataObj.M
    workersStartIdx = dataObj.M + 1
    workersEndIdx = dataObj.M + dataObj.N
    cap = np.zeros((numOfVertices, numOfVertices))
    cap_hat = []
    if (dataObj.MCMF_constraints_enabled == 1):
        energy_cap = np.zeros((numOfVertices, numOfVertices))
    if (dataObj.budget_con_enabled == 1):
        cap_hat = np.zeros((numOfVertices, numOfVertices))
    cost = np.zeros((numOfVertices, numOfVertices))
    
    ## Init cap matrix according to these cases:
    ## Case 1: init cap edges from src node to all tasks' nodes by the number of replicas of each task.
    total_replicas = np.sum(dataObj.optimal_num_of_replicas_closed_form)
    for j in range(0, dataObj.M):
        gammaReplicas_j = dataObj.optimal_num_of_replicas_closed_form[j]
        cap[srcNodeIdx][j + 1] = gammaReplicas_j
        if (dataObj.MCMF_constraints_enabled == 1):
            energy_cap[srcNodeIdx][j + 1] = np.max(dataObj.tasks_workers_comp_energies) * gammaReplicas_j
        if (dataObj.budget_con_enabled == 1):
            cap_hat[srcNodeIdx][j + 1] = dataObj.tasks_budgets[j]
    ## Case 2: init cap edges from tasks' nodes to workers' nodes by ones. 
    for j in range(0, dataObj.M):
        for i in range(dataObj.M + 1, numOfVertices - 1):
            if dataObj.MCMF_constraints_enabled == 1:
                if hasValidEdge(dataObj.tasks_workers_cons, i - (dataObj.M + 1), j):
                    cap[j + 1][i] = 1
                else:
                    cap[j + 1][i] = 0
            else:
                cap[j + 1][i] = 1

    if dataObj.MCMF_constraints_enabled == 1:
        ## Case 3: init energy_cap edges from tasks' nodes to workers' nodes by the energy consumption of each task on each worker.
        for j in range(0, dataObj.M):
            for i in range(dataObj.M + 1, numOfVertices - 1):
                energy_cap[j + 1][i] = dataObj.tasks_workers_comp_energies[j][i - (dataObj.M + 1)]

    if dataObj.budget_con_enabled == 1:
        ## Case 3: init cap_hat edges from tasks' nodes to workers' nodes by the workers' costs of each worker.
        for j in range(0, dataObj.M):
            for i in range(dataObj.M + 1, numOfVertices - 1):
                cap_hat[j + 1][i] = dataObj.workers_fitness_costs[i - (dataObj.M + 1)]
                # if isTaskBudgetMatchesWorkerFitnessCost(dataObj.tasks_budgets[j],
                #                                          dataObj.workers_fitness_costs[i - (dataObj.M + 1)]):
                #     cap_hat[j + 1][i] = dataObj.workers_fitness_costs[i - (dataObj.M + 1)]
                # else:
                #     cap_hat[j + 1][i] = 0
    
    ## Case 4: init cap edges from workers' nodes to sink node by the maximum number of tasks that each task can carry on. 
    for i in range(dataObj.M + 1, numOfVertices - 1):
        wTasks_i = dataObj.workers_max_tasks[i - (dataObj.M + 1)]
        cap[i][sinkNodeIdx] = wTasks_i
    
    if dataObj.MCMF_constraints_enabled == 1:
        ## Case 5: init energy_cap edges from workers' nodes to sink node by the maximum energy that each worker can carry on.  
        for i in range(dataObj.M + 1, numOfVertices - 1):
            wEnergy_i = dataObj.max_energies[i - (dataObj.M + 1)]
            energy_cap[i][sinkNodeIdx] = wEnergy_i

    if dataObj.budget_con_enabled == 1:
        ## Case 5: init cap_hat edges from workers' nodes to sink node by 1 
        # because if a worker is recruited once to a task, it can't be recruited again to the another task.
        for i in range(dataObj.M + 1, numOfVertices - 1):
            cap_hat[i][sinkNodeIdx] = float('inf')
    ## Case 6: 0 otherwise


    ## init cost matrix according to these cases:
    ## Case 1: from tasks' nodes to workers' nodes, apply positive costs
    #  on the forward edges and negative costs on the backward edges
    dataObj.objectiveFunction_inv = np.reshape(dataObj.objectiveFunction_inv, (dataObj.N, dataObj.M))

    for j in range(0, dataObj.M):
        for i in range(dataObj.M + 1, numOfVertices - 1):
            if dataObj.MCMF_constraints_enabled == 1:
                if hasValidEdge(dataObj.tasks_workers_cons, i - (dataObj.M + 1), j):
                    worker_i_task_j_cost = dataObj.objectiveFunction_inv[i - (dataObj.M + 1)][j]
                    cost[j + 1][i] = worker_i_task_j_cost
                    cost[i][j + 1] = -1 * worker_i_task_j_cost
                else:
                    cost[j + 1][i] = 0
            else:
                worker_i_task_j_cost = dataObj.objectiveFunction_inv[i - (dataObj.M + 1)][j]
                cost[j + 1][i] = worker_i_task_j_cost
                cost[i][j + 1] = -1 * worker_i_task_j_cost
    
    #Visualize capacity and cost graphs
    # G = GV.GraphVisualization()
    # G.buildGraphFromAdjMatrix(cap, 'capacity graph')

    # G.buildGraphFromAdjMatrix(cost, 'cost graph')
    # G.showGraphs()
    startTime = timeit.default_timer()
    while (True):
        result = bellmanFord_shortestPath(numOfVertices, srcNodeIdx, cap = cap,tasksNodesStartIdx=tasksStartIdx,
                                           tasksNodesEndIdx=tasksEndIdx,workersNodesStartIdx=workersStartIdx, workersNodesEndIdx=workersEndIdx,
                                           cost = cost, budget_con_enabled=dataObj.budget_con_enabled, 
                                           cap_hat=cap_hat)
        distance = result[0]
        p = result[1]
        if distance[sinkNodeIdx] == float('inf'):
            break
        #s = np.count_nonzero(p > -1) - 1
        ## Find the minimum flow along the found path
        cur = sinkNodeIdx;
        f = float('inf')
        f_energy = float('inf')
        f_budget = float('inf')
        while (cur != srcNodeIdx):
            f = min(f, cap[p[cur]][cur])
            if dataObj.MCMF_constraints_enabled == 1:
                f_energy = min(f_energy, energy_cap[p[cur]][cur])
            if dataObj.budget_con_enabled == 1:
                f_budget = min(f_budget, cap_hat[p[cur]][cur])
            cur = p[cur]
        

        #Update the capacities along the found path
        cur = sinkNodeIdx;
        assignedTaskNodeIdx = -1
        #cost[p[cur]][cur] = cost[p[cur]][cur] + (alpha * cost[p[p[cur]]][p[cur]])
        while (cur != srcNodeIdx):
            if cur >= tasksStartIdx and cur <= tasksEndIdx:
                assignedTaskNodeIdx = cur
            cap[p[cur]][cur] -= f
            cap[cur][p[cur]] += f
            if dataObj.MCMF_constraints_enabled == 1:
                energy_cap[p[cur]][cur] -= f_energy
                energy_cap[cur][p[cur]] += f_energy
            if dataObj.budget_con_enabled == 1:
                cap_hat[p[cur]][cur] -= f_budget
                cap_hat[cur][p[cur]] += f_budget
            cur = p[cur];
        if dataObj.budget_con_enabled == 1:
            #Get the priority level associated with the assigned task
            assignedTaskPriorityLevel = dataObj.tasks_priorities[assignedTaskNodeIdx - tasksStartIdx]
            #Find all tasks indices with the same priority level
            tasksIndicesWithSamePriorityLevel = np.where(dataObj.tasks_priorities == assignedTaskPriorityLevel)[0]
            if assignedTaskNodeIdx != -1:
                for taskIdx in tasksIndicesWithSamePriorityLevel:
                    #Update all budget limits of all edged between the source node and tasks nodes,
                    #  except assignedTaskNodeIdx
                    if taskIdx != (assignedTaskNodeIdx - tasksStartIdx):
                        cap_hat[srcNodeIdx][taskIdx + tasksStartIdx] = cap_hat[srcNodeIdx][assignedTaskNodeIdx]
        
                

    ## Return assignments
    i = workersStartIdx
    j = tasksStartIdx
    workersTasksAssignments = [ [] for i in range(dataObj.N)]
    X = np.zeros((dataObj.N * dataObj.M))
    totalNumberOfAllocatedWorkers = 0
    for i in range(workersStartIdx, workersStartIdx + dataObj.N):
        worker_idx = i - workersStartIdx
        for j in range(tasksStartIdx, dataObj.M + 1):
            task_idx = j - tasksStartIdx
            if (cap[i][j] == 1):
                workersTasksAssignments[i - workersStartIdx].append((j - tasksStartIdx) + 1)
                totalNumberOfAllocatedWorkers += 1
                X[(worker_idx * dataObj.M) + task_idx] = 1

    stopTime = timeit.default_timer()
    dataObj.MCMF_result = scratch.Scratch()
    dataObj.MCMF_result.workersTasksAssignments = workersTasksAssignments
    dataObj.MCMF_result.X = X
    dataObj.MCMF_result.runtime = stopTime - startTime
    dataObj.MCMF_result.totalNumberOfAllocatedWorkers = totalNumberOfAllocatedWorkers
    dataObj = MCMF_get_stats(dataObj)
    return dataObj.MCMF_result

def MCMF_get_stats(dataObj):
    return dataObj

def hasValidEdge(edgesLinks, i, j):
    return edgesLinks[i][j] == 1

def isTaskBudgetMatchesWorkerFitnessCost(task_budget, worker_fitness_cost):
    return task_budget >= worker_fitness_cost

def MCMF_preprocessing(dataObj):
    # 1. Create a mutlti-dimensional array of size [N][M]. 
    # This array will contain values of 0 or 1. 1 if the task can run on the worker and 0 otherwise according to the constraints.
    dataObj.workers_tasks_comp_energies = np.reshape(dataObj.comp_energies, (dataObj.N, dataObj.M))
    dataObj.tasks_workers_comp_energies = np.reshape(dataObj.comp_energies, (dataObj.N, dataObj.M)).transpose()
    dataObj.tasks_workers_cons = np.zeros((dataObj.N, dataObj.M))
    for i in range(dataObj.N):
        for j in range(dataObj.M):
            idx = (i * dataObj.M) + j
            #Check constraint b
            if (dataObj.tasks_comm_delays[idx] > dataObj.tasks_deadlines[j]): 
                dataObj.tasks_workers_cons[i][j] = 0
                continue
            else:
                dataObj.tasks_workers_cons[i][j] = 1

            #Check constraint c
            if (dataObj.workers_tasks_comp_energies[i][j] > dataObj.max_energies[i]):
                dataObj.tasks_workers_cons[i][j] = 0
                continue
            else: 
                dataObj.tasks_workers_cons[i][j] = 1

            #Check constraint e
            if (dataObj.tasks_execution_times[idx] > dataObj.tasks_deadlines[j]): 
                dataObj.tasks_workers_cons[i][j] = 0
                continue
            else:
                dataObj.tasks_workers_cons[i][j] = 1

    return dataObj
    


