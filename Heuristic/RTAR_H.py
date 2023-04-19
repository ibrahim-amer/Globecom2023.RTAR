# Define a function to perform the deferred acceptance algorithm
import numpy as np
def deferred_acceptance(college_prefs, student_prefs):
    # Initialize each college's list of accepted students to be empty
    college_accepted = {college: [] for college in college_prefs.keys()}
    # Initialize each student's match to be None
    student_match = {student: None for student in student_prefs.keys()}

    # Repeat until all students have been matched
    while None in student_match.values():
        # Each student applies to their most preferred college
        for student in student_match.keys():
            if student_match[student] is None:
                college = student_prefs[student].pop(0)
                # The college considers all its applicants
                college_applicants = college_accepted[college] + [student]
                # If the number of applicants is less than or equal to the number of slots, accept all applicants
                if len(college_applicants) <= len(college_prefs[college]):
                    college_accepted[college] = college_applicants
                    for applicant in college_applicants:
                        student_match[applicant] = college
                # Otherwise, accept only the most preferred applicants up to the number of slots
                else:
                    college_slots = len(college_prefs[college])
                    college_accepted[college] = college_prefs[college][:college_slots]
                    for applicant in college_accepted[college]:
                        student_match[applicant] = college

    # Return the final matching
    return student_match

def RTAR_H(tasks_prefs, tasks_budgets, workers_recruitment_costs) -> dict:
    # this function is used to match a set of tasks to a set of workers. 
    # each worker has a recruitment cost and each task has a budget limit. 
    # each task can be assigned to multiple workers but each worker can only be assigned to one task.
    # the input to this function is a dictionary of tasks preferences
    # the output is a dictionary of tasks assignments to workers

    #Convert inputs to np.array
    tasks_budgets = np.array(tasks_budgets)
    workers_recruitment_costs = np.array(workers_recruitment_costs)
    current_tasks_budgets = tasks_budgets.copy()
    available_workers = np.arange(len(workers_recruitment_costs))
    N = len(workers_recruitment_costs) # number of workers
    M = len(tasks_budgets) # number of tasks

    # initialize the dictionary of tasks assignments to workers
    tasks_assignments = {task: [] for task in tasks_prefs.keys()} 

    while ((current_tasks_budgets > 0).any() 
           and np.array([len(a) for a in tasks_prefs.values()]).any()
           and (available_workers.size > 0)): 
        # Each task applies to its most preferred worker
        for task in tasks_assignments.keys():
            #if current_tasks_budgets[task] > 0:
            worker = tasks_prefs[task][0]
            tasks_prefs[task] = np.delete(tasks_prefs[task], 0)
            if (worker in available_workers) and \
                    (current_tasks_budgets[task] >= workers_recruitment_costs[worker]):
                tasks_assignments[task].append(worker)
                current_tasks_budgets[task] -= workers_recruitment_costs[worker]
                available_workers = np.delete(available_workers, np.where(available_workers == worker))
            else:
                #do nothing
                pass
    return tasks_assignments

def convert_matchings_to_vector(matchings, N, M):
    # this function is used to convert the output of RTAR_H function to a vector of assignments
    # the output is a vector of assignments
    #M is the number of tasks
    #N is the number of workers
    vector = np.zeros(N * M)
    for task in matchings.keys():
        for worker in matchings[task]:
            idx = (worker * M) + task
            vector[idx] = 1
    return vector

def run_RTAR_H(tasks_budgets, workers_recruitment_costs, workers_reputations):
    #this function prepare the input for RTAR_H method
    #Basically, it creates a dictionary of tasks preferences by sorting the workers based on their reputations

    tasks_prefs = {}
    for task in range(len(tasks_budgets)):
        tasks_prefs[task] = np.argsort(workers_reputations)[::-1]
    matchings = RTAR_H(tasks_prefs, tasks_budgets, workers_recruitment_costs)
    return matchings

def run_RTAR_H_vectorized(tasks_budgets, workers_recruitment_costs, workers_reputations):
    #this function prepare the input for RTAR_H method
    #Basically, it creates a dictionary of tasks preferences by sorting the workers based on their reputations

    tasks_prefs = {}
    for task in range(len(tasks_budgets)):
        tasks_prefs[task] = np.argsort(workers_reputations)[::-1]
    matchings = RTAR_H(tasks_prefs, tasks_budgets, workers_recruitment_costs)
    X = convert_matchings_to_vector(matchings, len(workers_recruitment_costs), len(tasks_budgets))
    return X



