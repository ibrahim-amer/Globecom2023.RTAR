function [] = reload_python_workspace(env)
%env = 1: Local machine
% env = 2: Queen's remote machine
%Unload Module
%clear classes
RTAR_H_path = [];
if (env == 1)
    RTAR_H_path = 'C:\Ibrahim-Workspace-Main\Research\Queen''s University\PhD\Research\Dr. Sharief & Dr. Hossam\Repos\Globecom2023.RTAR\Heuristic';
elseif (env == 2)
    RTAR_H_path = 'C:\Users\ibrahim.amer\Ibrahim-Workspace\Repos\Task.Replication\PBTA.Simulation\Greedy.MCMF';
end
module_name = 'RTAR_H';
if count(py.sys.path, RTAR_H_path) == 0
    insert(py.sys.path,int32(0), RTAR_H_path);
end
mod = py.importlib.import_module(module_name);
py.importlib.reload(mod);

end



