#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 15:47, 06/01/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

''' Formular for cost constraint:
    Constraint: es * (E(rp + rs) +  E(qp + qs)) < cost_max
    From: es * (E(rp + rs) + E(qp + qs)) = mean_cost
    => estimate_param (es) = (gamma * mean_cost - E(qp + qs)) / (E(rp + rs) - E(qp + qs))
'''

from model.scheduler.schedule import Schedule
from utils.ops_util import find_closet_node, is_instance_of_fog, get_cloud_node


def get_punished_cost(exceeded_cost):
    return 0.5 * exceeded_cost


def get_penaty_forwarding_cost(clouds: {}, fogs: {}, peers: {}, tasks: {}, schedule: Schedule, estimate_param) -> float:
    exceeded_cost = 0

    for time_slot in range(schedule.total_time):
        for fog_id, list_task_id in schedule.schedule_fogs_tasks.items():
            if len(list_task_id) > time_slot:
                task = tasks[list_task_id[time_slot]]
                exceeded_cost += max(0, (estimate_param * (task.d_p + task.d_s) - task.cost_max) ** 2)

        for cloud_id, list_task_id in schedule.schedule_clouds_tasks.items():
            if len(list_task_id) > time_slot:
                task_id = list_task_id[time_slot]
                task = tasks[task_id]
                exceeded_cost += max(0, (estimate_param * (task.d_p + task.d_s) - task.cost_max) ** 2)

    for peer_id, list_task_id in schedule.schedule_peers_tasks.items():
        for task_id in list_task_id:
            task = tasks[task_id]
            exceeded_cost += max(0, (estimate_param * (task.d_p + task.d_s) - task.cost_max) ** 2)
            exceeded_cost += max(0, (estimate_param * (task.d_p + task.d_p) - task.cost_max) ** 2)
    
    return get_punished_cost(exceeded_cost)


def data_forwarding_cost(clouds: {}, fogs: {}, peers: {}, tasks: {}, schedule: Schedule) -> float:
    cloud_cost = 0
    fog_cost = 0
    peer_cost = 0

    total_rps = 0
    total_qps = 0
    n_rps = 0
    n_qps = 0

    for time_slot in range(schedule.total_time):
        for fog_id, list_task_id in schedule.schedule_fogs_tasks.items():
            fog = fogs[fog_id]
            fog_cost += fog.sigma_device_idle + fog.sigma_idle
            if len(list_task_id) > time_slot:
                task = tasks[list_task_id[time_slot]]
                fog_cost += (fog.sigma_device + fog.sigma) * (task.d_p + task.d_s)

                total_rps += (task.d_p + task.d_s)
                n_rps += 1

        for cloud_id, list_task_id in schedule.schedule_clouds_tasks.items():
            if len(list_task_id) <= time_slot:
                continue
            task_id = list_task_id[time_slot]
            task = tasks[task_id]
            cloud = clouds[cloud_id]
            fog = find_closet_node(cloud, fogs)
            cloud_cost += fog.sigma_device_idle + fog.sigma_idle + cloud.sigma_idle + (fog.sigma_device + fog.sigma + cloud.sigma) * (task.d_p + task.d_s)
            total_qps += (task.d_p + task.d_s)
            n_qps += 1

    # We have already calculated the cost to transfer data from IoTs to fog or cloud, we only need to calculate cost to transfer from fog/cloud to blockchain only
    list_task_handler = schedule.get_list_task_handler()
    for peer_id, list_task_id in schedule.schedule_peers_tasks.items():
        for task_id in list_task_id:
            peer = peers[peer_id]
            task = tasks[task_id]
            node_id = list_task_handler[task_id]
            fog = is_instance_of_fog(fogs, node_id)
            if fog is None:  # Cloud
                cloud = get_cloud_node(clouds, node_id)
                peer_cost += (cloud.sigma_idle + peer.sigma_sm) + (cloud.sigma + peer.sigma) * (task.d_p + task.d_s)
                total_qps += (task.d_p + task.d_s)
                n_qps += 1
            else:       # Fog
                peer_cost += (fog.sigma_idle + peer.sigma_sm) +  (fog.sigma + peer.sigma) * (task.d_p + task.d_s)
                total_rps += (task.d_p + task.d_s)
                n_rps += 1
      
    mean_cost = (cloud_cost + fog_cost + peer_cost) / schedule.n_tasks
    mean_rps = mean_qps = 0
    if n_qps == 0 and n_rps == 0:
        return cloud_cost + fog_cost + peer_cost
    else:
        if n_qps != 0 and n_rps != 0:
            estimate_param = mean_cost / (total_qps / n_qps + total_rps / n_rps)
        elif n_qps != 0:
            estimate_param = mean_cost / (total_qps / n_qps)
        elif n_rps != 0:
            estimate_param = mean_cost / (total_rps / n_rps)
        punished_cost = get_penaty_forwarding_cost(clouds, fogs, peers, tasks, schedule, estimate_param)
        # print(alpha)
        # print(mean_cost, mean_qps, mean_rps)
        # print([cloud_cost + fog_cost + peer_cost, punished_cost])
        # print([cloud_cost + fog_cost + peer_cost, punished_cost])
        # print(punished_cost)
        return cloud_cost + fog_cost + peer_cost + punished_cost


def computation_cost(clouds: {}, fogs: {}, peers: {}, tasks: {}, schedule: Schedule) -> float:
    cloud_cost = 0
    fog_cost = 0

    for time_slot in range(schedule.total_time):
        for fog_id, list_task_id in schedule.schedule_fogs_tasks.items():
            fog = fogs[fog_id]
            fog_cost += fog.pi_idle
            if len(list_task_id) > time_slot:
                task = tasks[list_task_id[time_slot]]
                fog_cost += fog.pi * task.d_p
                start_time_slot = max(0, time_slot - fog.tau)
                for j in range(start_time_slot, time_slot):
                    task = tasks[list_task_id[j]]
                    factor = 1 / ((time_slot - j) ** 2 + 1)
                    fog_cost += factor * fog.pi * task.d_s

        for cloud_id, list_task_id in schedule.schedule_clouds_tasks.items():
            cloud = clouds[cloud_id]
            cloud_cost += cloud.pi_idle
            if len(list_task_id) > time_slot:
                task = tasks[list_task_id[time_slot]]
                cloud_cost += cloud.pi * task.d_p
                for j in range(time_slot):
                    task = tasks[list_task_id[j]]
                    factor = 1 / ((time_slot - j) ** 2 + 1)
                    cloud_cost += factor * cloud.pi * task.d_s

    return cloud_cost + fog_cost


def storage_cost(clouds: {}, fogs: {}, peers: {}, tasks: {}, schedule: Schedule) -> float:
    cloud_cost = 0
    fog_cost = 0
    peer_cost = 0

    for time_slot in range(schedule.total_time):
        for cloud_id, list_task_id in schedule.schedule_clouds_tasks.items():
            cloud = clouds[cloud_id]
            cloud_cost += cloud.omega_idle
            for j in range(time_slot):
                if len(list_task_id) > j:
                    task = tasks[list_task_id[j]]
                    cloud_cost += cloud.omega * task.d_s

        for fog_id, list_task_id in schedule.schedule_fogs_tasks.items():
            fog = fogs[fog_id]
            fog_cost += fog.omega_idle
            start_time_slot = max(0, time_slot - fog.tau)
            for j in range(start_time_slot, time_slot):
                if len(list_task_id) > j:
                    task = tasks[list_task_id[j]]
                    fog_cost += fog.omega * task.d_s

    for peer_id, list_task_id in schedule.schedule_peers_tasks.items():
        for task_id in list_task_id:
            peer = peers[peer_id]
            task = tasks[task_id]
            peer_cost += peer.omega_sm + peer.omega * (task.d_s + task.d_p)

    return cloud_cost + fog_cost + peer_cost