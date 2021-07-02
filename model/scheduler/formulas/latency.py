#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 15:48, 06/01/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from model.scheduler.schedule import Schedule
from utils.ops_util import find_closet_node


def transmission_latency(clouds: {}, fogs: {}, peers: {}, tasks: {}, schedule: Schedule) -> float:
    cloud_latency = 0
    fog_latency = 0

    for fog_id, list_task_id in schedule.schedule_fogs_tasks.items():
        for task_id in list_task_id:
            fog = fogs[fog_id]
            task = tasks[task_id]
            fog_latency += fog.eta * (task.d_p + task.d_s)

    for cloud_id, list_task_id in schedule.schedule_clouds_tasks.items():
        for task_id in list_task_id:
            task = tasks[task_id]
            cloud = clouds[cloud_id]
            fog = find_closet_node(cloud, fogs)
            cloud_latency += (fog.eta + cloud.eta) * (task.d_p + task.d_s)
    # temp = cloud_latency + fog_latency
    # print(f"Fog: {fog_latency}, Cloud: {cloud_latency}")
    return cloud_latency + fog_latency


def processing_latency(clouds: {}, fogs: {}, peers: {}, tasks: {}, schedule: Schedule) -> float:
    cloud_latency = 0
    fog_latency = 0

    for cloud_id, list_task_id in schedule.schedule_clouds_tasks.items():
        cloud = clouds[cloud_id]
        for time_slot, task_id in enumerate(list_task_id):
            task = tasks[task_id]
            cloud_latency += cloud.lamda * task.d_p
            for j in range(time_slot):
                task = tasks[list_task_id[j]]
                factor = 1 / ((time_slot - j) ** 2 + 1)
                cloud_latency += factor * cloud.lamda * task.d_s

    for fog_id, list_task_id in schedule.schedule_fogs_tasks.items():
        fog = fogs[fog_id]
        for time_slot, task_id in enumerate(list_task_id):
            task = tasks[task_id]
            fog_latency += fog.lamda * task.d_p
            start_time_slot = max(0, time_slot - fog.tau)
            for j in range(start_time_slot, time_slot):
                task = tasks[list_task_id[j]]
                factor = 1 / ((time_slot - j) ** 2 + 1)
                fog_latency += factor * fog.lamda * task.d_s

    # print(f"Fog: {fog_latency}, Cloud: {cloud_latency}")
    return cloud_latency + fog_latency


def punishment_latency(clouds: {}, fogs: {}, peers: {}, tasks: {}, schedule: Schedule, trans_lat:float, pro_lat:float) -> float:

    def punish_func(delay):
        if delay <= 0:
            return 0
        elif 0 < delay <= 1:
            return delay
        else:
            return (delay ** 2 + 1)/2

    # task_latency_dict = {}
    # ## Calculate transmission latency for single task
    # task_handler_dict = schedule.get_list_task_handler()  # { taskId: nodeID }
    # for task_id, node_id in task_handler_dict.items():
    #     task = tasks[task_id]
    #     fog = is_instance_of_fog(fogs, node_id)
    #     latency_time = 0
    #     if fog is None:
    #         cloud = get_cloud_node(clouds, node_id)
    #         fog = find_closet_node(cloud, fogs)
    #         latency_time += fog.eta * (task.d_p + task.d_s) + (fog.eta + cloud.eta) * (task.d_p + task.d_s)
    #     else:
    #         latency_time += fog.eta + (task.d_p + task.d_s)
    #     task_latency_dict[task_id] = latency_time
    #
    # ## Calculate processing latency for single task
    # task_handlers_order_dict = schedule.get_list_task_handler_with_order()  # { taskId: { fogID: [], CloudId: [] } }
    # for task_id, handlers in task_handlers_order_dict.items():
    #     task = tasks[task_id]
    #     task_latency = 0
    #     for idx, (handler_id, list_task_before) in enumerate(handlers.items()):
    #         if idx == 0:
    #             fog = fogs[handler_id]
    #             task_latency += fog.lamda * task.d_p
    #             start_time = fog.tau - len(list_task_before)
    #             if start_time > 0 and len(list_task_before) != 0:
    #                 factor = 1 / (start_time ** 2 + 1)
    #                 task_latency += factor * fog.lamda * task.d_s
    #         else:
    #             cloud = clouds[handler_id]
    #             task_latency += cloud.lamda * task.d_p
    #             start_time = len(list_task_before)
    #             factor = 1 / (start_time ** 2 + 1)
    #             task_latency += factor * cloud.lamda * task.d_s
    #     task_latency_dict[task_id] += task_latency

    total = trans_lat + pro_lat - sum([task.sl_max for task in tasks.values()])
    total_punish = punish_func(total)

    # print(f"Punish: {total_punish}, Total: {total}")
    return total_punish


    ## Calculate total punishment
    # total_punish = [punish_func(task_latency - tasks[task_id].sl_max) for task_id, task_latency in task_latency_dict.items()]
    # temp = sum(total_punish)
    # print(f"Punish: {temp}")
    # return sum(total_punish)
