#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 15:47, 06/01/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from model.scheduler.schedule import Schedule
from utils.ops_util import find_closet_node, is_instance_of_fog, get_cloud_node


def data_forwarding_cost(clouds: {}, fogs: {}, peers: {}, tasks: {}, schedule: Schedule) -> float:
    cloud_cost = 0
    fog_cost = 0
    peer_cost = 0

    total_dps = 0
    n_dps = 0

    for time_slot in range(schedule.total_time):
        for fog_id, list_task_id in schedule.schedule_fogs_tasks.items():
            fog = fogs[fog_id]
            fog_cost += fog.sigma_device_idle + fog.sigma_idle
            if len(list_task_id) > time_slot:
                task = tasks[list_task_id[time_slot]]
                change = (fog.sigma_device + fog.sigma) * (task.d_p + task.d_s)
                fog_cost += change
                task.total_cost += change
                n_dps += 1

        for cloud_id, list_task_id in schedule.schedule_clouds_tasks.items():
            if len(list_task_id) <= time_slot:
                continue
            task_id = list_task_id[time_slot]
            task = tasks[task_id]
            cloud = clouds[cloud_id]
            fog = find_closet_node(cloud, fogs)
            change = fog.sigma_device_idle + fog.sigma_idle + cloud.sigma_idle + (fog.sigma_device + fog.sigma + cloud.sigma) * (task.d_p + task.d_s)
            cloud_cost += change
            task.total_cost += change
            n_dps += 1

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
                change = (cloud.sigma_idle + peer.sigma_sm) + (cloud.sigma + peer.sigma) * (task.d_p + task.d_s)
                peer_cost += change
                task.total_cost += change
                n_dps += 1
            else:  # Fog
                change = (fog.sigma_idle + peer.sigma_sm) +  (fog.sigma + peer.sigma) * (task.d_p + task.d_s)
                peer_cost += change
                task.total_cost += change
                total_dps += (task.d_p + task.d_s)
                n_dps += 1
    
    return cloud_cost + fog_cost + peer_cost # + punished_cost


def computation_cost(clouds: {}, fogs: {}, peers: {}, tasks: {}, schedule: Schedule) -> float:
    cloud_cost = 0
    fog_cost = 0
    total_dps = 0
    n_dps = 0

    for time_slot in range(schedule.total_time):
        for fog_id, list_task_id in schedule.schedule_fogs_tasks.items():
            fog = fogs[fog_id]
            fog_cost += fog.pi_idle
            if len(list_task_id) > time_slot:
                task = tasks[list_task_id[time_slot]]
                fog_cost += fog.pi * task.d_p
                total_dps += task.d_p
                n_dps += 1
                start_time_slot = max(0, time_slot - fog.tau)
                for j in range(start_time_slot, time_slot):
                    task = tasks[list_task_id[j]]
                    factor = 1 / ((time_slot - j) ** 2 + 1)
                    change = factor * fog.pi * task.d_s
                    fog_cost += change
                    task.total_cost += change

        for cloud_id, list_task_id in schedule.schedule_clouds_tasks.items():
            cloud = clouds[cloud_id]
            cloud_cost += cloud.pi_idle
            if len(list_task_id) > time_slot:
                task = tasks[list_task_id[time_slot]]
                cloud_cost += cloud.pi * task.d_p
                total_dps += task.d_p
                n_dps += 1
                for j in range(time_slot):
                    task = tasks[list_task_id[j]]
                    factor = 1 / ((time_slot - j) ** 2 + 1)
                    change = factor * cloud.pi * task.d_s
                    cloud_cost += change
                    task.total_cost += change
                    
    return cloud_cost + fog_cost # + punished_cost
    
    
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
                    change = cloud.omega * task.d_s
                    cloud_cost += change
                    task.total_cost += change
                    
        for fog_id, list_task_id in schedule.schedule_fogs_tasks.items():
            fog = fogs[fog_id]
            fog_cost += fog.omega_idle
            start_time_slot = max(0, time_slot - fog.tau)
            for j in range(start_time_slot, time_slot):
                if len(list_task_id) > j:
                    task = tasks[list_task_id[j]]
                    change = fog.omega * task.d_s
                    fog_cost += change
                    task.total_cost += change
                    
    for peer_id, list_task_id in schedule.schedule_peers_tasks.items():
        for task_id in list_task_id:
            peer = peers[peer_id]
            task = tasks[task_id]
            change = peer.omega_sm + peer.omega * (task.d_s + task.d_p)
            peer_cost += change
            task.total_cost += change
            
    return cloud_cost + fog_cost + peer_cost