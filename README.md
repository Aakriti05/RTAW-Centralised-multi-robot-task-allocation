# RTAW

1. For 100 robots on 500 tasks for all 5 layout go into folder "complex_task_100rob_layout_newtask"
	Testing - eg. for the Layout A
		pip install -e .
		cd warehouse
		python main_pytorch.py --load ./results/20220108T070145.145997/best/ --demo   # to get TTD value for RTAW (our method)
		python greedy.py # to get TTD value for greedy and regret baseline
		
	Training - eg. for layout A.
		Uncomment line "output = SoftmaxCategoricalHead()(output)" in network_policy.py code.
		python main_pytorch.py
		
	Note: To train or test a different layout change the layout name in the file, train it and then test it. Contact author for details.
	
2. For 10 robots on 500 tasks for all 5 layout go into folder "complex_task_10rob_layout_newtask" and follow similar steps as above.
	Testing -
		... 
		python main_pytorch.py --load ./results/20220110T012756.969029/best/ --demo
		...
	Training - 
		...

3. Code for 500/1000 robots with ORCA can be provided by author if asked. The code is similar to the above with changes in the robot/task number and navigation scheme respectively.

4. We have also provided the simple experiment code present in complex_task_10robots and complex_task_100robots.

