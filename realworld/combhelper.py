import networkx as nx
import cplex
import time
import pickle
import random
import logging
import sys
import itertools


logging.basicConfig(
    format='%(message)s',
    level=logging.DEBUG,
)


def base_mvc(G, alg):
    if alg == 'LP':
        prob = cplex.Cplex()
        prob.set_problem_name('MVC')
        prob.set_problem_type(prob.problem_type.MILP)
        prob.parameters.timelimit.set(3600.0)
        prob.objective.set_sense(prob.objective.sense.minimize)
        
        node_name = [str(i) for i in range(len(G))]
        
        prob.variables.add(names=node_name,
                        obj=[1]*len(node_name),
                        lb=[0]*len(node_name),
                        ub=[1]*len(node_name),
                        types=['I']*len(node_name))
        
        constraints = [[[str(u), str(v)], [1, 1]] for u, v in nx.edges(G)]
        
        constraints_names = [','.join(constraint[0]) for constraint in constraints]
        
        prob.linear_constraints.add(lin_expr=constraints,
                                    senses=['G']*len(constraints),
                                    rhs=[1]*len(constraints),
                                    names=constraints_names)
        
        t_start = prob.get_time()
        prob.solve()
        t_end = prob.get_time()
        solution_set = [int(node_name[i]) for i in range(len(node_name)) if prob.solution.get_values(i) == 1.0]
    
    elif alg == 'GD':
        G_copy = G.copy()
        solution_set = set()
        t_start = time.time()
        while nx.number_of_edges(G_copy):
            select_node = -1
            max_degree = -1
            for node in nx.nodes(G_copy):
                if nx.degree(G_copy, node) > max_degree:
                    select_node = node
                    max_degree = nx.degree(G_copy, node)
            solution_set.add(select_node)
            G_copy.remove_node(select_node)
        t_end = time.time()
    
    elif alg == 'LS':
        G_copy = G.copy()
        init_solution = set()
        while nx.number_of_edges(G_copy):
            u, v = random.choice([edge for edge in nx.edges(G_copy)])
            init_solution.add(u)
            init_solution.add(v)
            G_copy.remove_node(u)
            G_copy.remove_node(v)
        
        solution_set = init_solution.copy()
        best = len(solution_set)
        t_start = time.time()
        while True:
            for node in solution_set:
                cur_solution = solution_set.copy()
                cur_solution.remove(node)
                change = False
                
                nbrs = set(nx.neighbors(G, node))
                for nbr in nbrs:
                    if nbr not in cur_solution:
                        change = True
                        break

                if not change:
                    solution_set = cur_solution
                    break
            cur = len(solution_set)
            if cur < best:
                best = cur
            else:
                break
        t_end = time.time()
        
    logging.info('Time elapsed: {:.3f}s'.format(t_end - t_start))
    logging.info('Solution size: {:d}'.format(len(solution_set)))
    return solution_set


def our_mvc(G, preds, alg):
    if alg == 'LP':
        prob = cplex.Cplex()
        prob.set_problem_name('MVC')
        prob.set_problem_type(prob.problem_type.MILP)
        prob.parameters.timelimit.set(3600.0)
        prob.objective.set_sense(prob.objective.sense.minimize)
        
        node_name = [str(i) for i in range(len(G))]
        good_node = [i for i in range(len(preds)) if preds[i] == 1]
        
        prob.variables.add(names=node_name,
                        obj=[1]*len(node_name),
                        lb=[0]*len(node_name),
                        ub=preds,
                        types=['I']*len(node_name))
        
        constraints = [[[str(u), str(v)], [1, 1]] for u, v in nx.edges(G) if u in good_node or v in good_node]
        
        constraints_names = [','.join(constraint[0]) for constraint in constraints]
        
        prob.linear_constraints.add(lin_expr=constraints,
                                    senses=['G']*len(constraints),
                                    rhs=[1]*len(constraints),
                                    names=constraints_names)
        
        t_start = prob.get_time()
        prob.solve()
        t_end = prob.get_time()
        solution_set = [int(node_name[i]) for i in range(len(node_name)) if prob.solution.get_values(i) == 1.0]
    
    elif alg == 'GD':
        G_copy = G.copy()
        good_nodes = set(i for i in range(len(preds)) if preds[i] == 1)
        solution_set = set()
        t_start = time.time()
        while good_nodes:
            select_node = -1
            max_degree = -1
            for node in good_nodes:
                if nx.degree(G_copy, node) > max_degree:
                    select_node = node
                    max_degree = nx.degree(G_copy, node)
            if nx.degree(G_copy, select_node) == 0:
                break
            solution_set.add(select_node)
            good_nodes.remove(select_node)
            G_copy.remove_node(select_node)
        t_end = time.time()
        
    elif alg == 'LS':
        good_nodes = set(i for i in range(len(preds)) if preds[i] == 1)
        init_solution = good_nodes
        solution_set = init_solution.copy()
        best = len(solution_set)
        t_start = time.time()
        while True:
            for node in solution_set:
                cur_solution = solution_set.copy()
                cur_solution.remove(node)
                change = False
                
                nbrs = set(nx.neighbors(G, node))
                for nbr in nbrs:
                    if nbr not in cur_solution:
                        change = True
                        break

                if not change:
                    solution_set = cur_solution
                    break
            cur = len(solution_set)
            if cur < best:
                best = cur
            else:
                break
        t_end = time.time()
    
    logging.info('Time elapsed: {:.3f}s'.format(t_end - t_start))
    logging.info('Solution size: {:d}'.format(len(solution_set)))
    
    total_edges = nx.number_of_edges(G)
    G_copy = G.copy()
    covered_edges = 0
    for node in solution_set:
        covered_edges += nx.degree(G_copy, node)
        G_copy.remove_node(node)
    logging.info('Coverage: {:.4f}'.format(covered_edges / total_edges))
    return solution_set


def base_mis(G, alg):
    if alg == 'LP':
        prob = cplex.Cplex()
        prob.set_problem_name('MIS')
        prob.set_problem_type(prob.problem_type.MILP)
        prob.parameters.timelimit.set(3600.0)
        prob.objective.set_sense(prob.objective.sense.maximize)
        
        node_name = [str(i) for i in range(len(G))]
        
        prob.variables.add(names=node_name,
                        obj=[1]*len(node_name),
                        lb=[0]*len(node_name),
                        ub=[1]*len(node_name),
                        types=['I']*len(node_name))
        
        constraints = [[[str(u), str(v)], [1, 1]] for u, v in nx.edges(G)]
        
        constraints_names = [','.join(constraint[0]) for constraint in constraints]
        
        prob.linear_constraints.add(lin_expr=constraints,
                                    senses=['L']*len(constraints),
                                    rhs=[1]*len(constraints),
                                    names=constraints_names)
        
        t_start = prob.get_time()
        prob.solve()
        t_end = prob.get_time()
        solution_set = [int(node_name[i]) for i in range(len(node_name)) if prob.solution.get_values(i) == 1.0]
    
    elif alg == 'GD':
        G_copy = G.copy()
        max_degree = max([deg[1] for deg in nx.degree(G)])
        solution_set = set()
        t_start = time.time()
        while nx.number_of_nodes(G_copy):
            select_node = -1
            min_degree = max_degree + 1
            for node in nx.nodes(G_copy):
                if nx.degree(G_copy, node) < min_degree:
                    select_node = node
                    min_degree = nx.degree(G_copy, node)
            solution_set.add(select_node)
            node_remove = set()
            node_remove.add(select_node)
            for nbr in nx.neighbors(G_copy, select_node):
                node_remove.add(nbr)
            for node in node_remove:
                G_copy.remove_node(node)
        t_end = time.time()
    
    elif alg == 'LS':
        G_copy = G.copy()
        init_solution = set()
        while nx.number_of_nodes(G_copy):
            select_node = random.choice([node for node in nx.nodes(G_copy)])
            init_solution.add(select_node)
            node_remove = set()
            node_remove.add(select_node)
            for nbr in nx.neighbors(G_copy, select_node):
                node_remove.add(nbr)
            for node in node_remove:
                G_copy.remove_node(node)
                
        solution_set = init_solution.copy()
        best = len(solution_set)
        
        t_start = time.time()
        while True:
            for node in solution_set:
                change = False
                nbrs = set(nbr for nbr in nx.neighbors(G, node) if nbr not in solution_set)
                if len(nbrs) >= 2:
                    node_pairs = set(itertools.combinations(nbrs, 2))
                    for u, v in node_pairs:
                        one_tight = True
                        for nbr in set(nx.neighbors(G, u)):
                            if nbr != node and nbr in solution_set:
                                one_tight = False
                                break
                            
                        for nbr in set(nx.neighbors(G, v)):
                            if nbr != node and nbr in solution_set:
                                one_tight = False
                                break
                        
                        if one_tight and ((u, v) not in nx.edges(G) and (v, u) not in nx.edges(G)):
                            solution_set.add(u)
                            solution_set.add(v)
                            solution_set.remove(node)
                            change = True
                            break
                if change:
                    break
            cur = len(solution_set)
            if cur > best:
                best = cur
            else:
                break
        t_end = time.time()
        
    logging.info('Time elapsed: {:.3f}s'.format(t_end - t_start))
    logging.info('Solution size: {:d}'.format(len(solution_set)))
    return solution_set


def our_mis(G, preds, alg):
    if alg == 'LP':
        prob = cplex.Cplex()
        prob.set_problem_name('MIS')
        prob.set_problem_type(prob.problem_type.MILP)
        prob.parameters.timelimit.set(3600.0)
        prob.objective.set_sense(prob.objective.sense.maximize)
        
        node_name = [str(i) for i in range(len(G))]
        good_node = [i for i in range(len(preds)) if preds[i] == 1]
        
        prob.variables.add(names=node_name,
                        obj=[1]*len(node_name),
                        lb=[0]*len(node_name),
                        ub=preds,
                        types=['I']*len(node_name))
        
        constraints = [[[str(u), str(v)], [1, 1]] for u, v in nx.edges(G) if u in good_node or v in good_node]
        
        constraints_names = [','.join(constraint[0]) for constraint in constraints]
        
        prob.linear_constraints.add(lin_expr=constraints,
                                    senses=['L']*len(constraints),
                                    rhs=[1]*len(constraints),
                                    names=constraints_names)
        
        t_start = prob.get_time()
        prob.solve()
        t_end = prob.get_time()
        solution_set = [int(node_name[i]) for i in range(len(node_name)) if prob.solution.get_values(i) == 1.0]
    
    elif alg == 'GD':
        G_copy = G.copy()
        good_nodes = set(i for i in range(len(preds)) if preds[i] == 1)
        max_degree = max([deg[1] for deg in nx.degree(G)])
        solution_set = set()
        t_start = time.time()
        while good_nodes:
            select_node = -1
            min_degree = max_degree + 1
            for node in good_nodes:
                if nx.degree(G_copy, node) < min_degree:
                    select_node = node
                    min_degree = nx.degree(G_copy, node)
            solution_set.add(select_node)
            remove_from_good_nodes = set()
            remove_from_good_nodes.add(select_node)
            remove_from_G_copy = set()
            remove_from_G_copy.add(select_node)
            for nbr in nx.neighbors(G_copy, select_node):
                remove_from_G_copy.add(nbr)
                if nbr in good_nodes:
                    remove_from_good_nodes.add(nbr)
            for node in remove_from_good_nodes:
                good_nodes.remove(node)
            for node in remove_from_G_copy:
                G_copy.remove_node(node)
        t_end = time.time()
        
    elif alg == 'LS':
        good_nodes = set(i for i in range(len(preds)) if preds[i] == 1)
        G_copy = G.copy()
        init_solution = set()
        while good_nodes:
            select_node = random.choice([node for node in good_nodes])
            init_solution.add(select_node)
            node_remove_from_good = set()
            node_remove_from_good.add(select_node)
            node_remove_from_G = set()
            node_remove_from_G.add(select_node)
            for nbr in nx.neighbors(G_copy, select_node):
                node_remove_from_G.add(nbr)
                if nbr in good_nodes:
                    node_remove_from_good.add(nbr)
            for node in node_remove_from_good:
                good_nodes.remove(node)
            for node in node_remove_from_G:
                G_copy.remove_node(node)
        
        solution_set = init_solution.copy()
        best = len(solution_set)
        t_start = time.time()
        while True:
            for node in solution_set:
                change = False
                nbrs = set(nbr for nbr in nx.neighbors(G, node) if (nbr not in solution_set) and (nbr in good_nodes))
                if len(nbrs) >= 2:
                    node_pairs = set(itertools.combinations(nbrs, 2))
                    for u, v in node_pairs:
                        one_tight = True
                        for nbr in set(nx.neighbors(G, u)):
                            if nbr != node and nbr in solution_set:
                                one_tight = False
                                break
                            
                        for nbr in set(nx.neighbors(G, v)):
                            if nbr != node and nbr in solution_set:
                                one_tight = False
                                break
                        if one_tight and ((u, v) not in nx.edges(G) and (v, u) not in nx.edges(G)):
                            solution_set.add(u)
                            solution_set.add(v)
                            solution_set.remove(node)
                            change = True
                            break
                if change:
                    break
            cur = len(solution_set)
            if cur > best:
                best = cur
            else:
                break
        t_end = time.time()
    
    logging.info('Time elapsed: {:.3f}s'.format(t_end - t_start))
    logging.info('Solution size: {:d}'.format(len(solution_set)))
    return solution_set


if __name__ == '__main__':
    random.seed(42)
    
    prob = sys.argv[1] # CO problems: MVC and MIS
    alg = sys.argv[2] # traditional CO algorithms: LP, GD and LS
    ds = sys.argv[3] # dataset name used in the paper, e.g., OTC, Pubmed, etc.
    
    # networkx graph instance
    G = pickle.load(open('./data/%s/raw/%s.G' % (ds, ds), 'rb'))
    
    if prob == 'MVC':
        logging.info('Problem: %s' % prob)
        
        # baseline
        logging.info('%s:' % alg)
        solution_base = base_mvc(G, alg)
        
        # combhepler_{pt}
        logging.info('%s+COMBHelper_{pt}:' % alg)
        preds_t = pickle.load(open('./preds/%s_MVC_Teacher.preds' % ds, 'rb'))['pred']
        solution_t = our_mvc(G, preds_t, alg)
        
        # combhepler
        logging.info('%s+COMBHelper:' % alg)
        preds_s = pickle.load(open('./preds/%s_MVC_Student.preds' % ds, 'rb'))['pred']
        solution_s = our_mvc(G, preds_s, alg)
        
    elif prob == 'MIS':
        logging.info('Problem: %s' % prob)
        
        # baseline
        logging.info('%s:' % alg)
        solution_base = base_mis(G, alg)
        
        # combhepler_{pt}
        logging.info('%s+COMBHelper_{pt}:' % alg)
        preds_t = pickle.load(open('./preds/%s_MIS_Teacher.preds' % ds, 'rb'))['pred']
        solution_t = our_mis(G, preds_t, alg)
        
        # combhepler
        logging.info('%s+COMBHelper:' % alg)
        preds_s = pickle.load(open('./preds/%s_MIS_Student.preds' % ds, 'rb'))['pred']
        solution_s = our_mis(G, preds_s, alg)