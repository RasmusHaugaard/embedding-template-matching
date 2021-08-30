import argparse

import rospy
import actionlib
import emb_template_ros.msg

parser = argparse.ArgumentParser()
parser.add_argument('server_name')
parser.add_argument('object_name')
args = parser.parse_args()

print('initializing node')
rospy.init_node('pose_client')
print('creating client')
client = actionlib.SimpleActionClient(f'{args.server_name}/getPose', emb_template_ros.msg.getPoseAction)
print('waiting for server')
client.wait_for_server()
print('sending goal')
client.send_goal(emb_template_ros.msg.getPoseGoal(object_name=args.object_name))
print('waiting for result')
client.wait_for_result()
print('getting result')
result = client.get_result()
print(result)
