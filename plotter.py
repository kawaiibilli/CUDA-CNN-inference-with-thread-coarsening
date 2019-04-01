import matplotlib.pyplot as plt


# conv1v1 = [7.940256,1.036352,1.332960,1.279968,1.251648,1.317952,1.369728,1.527360,1.731104,1.780672,1.947168,2.118336,2.283744,2.277056,2.425952,2.569728]
# conv1v2 = []


# granularity = []
# for i in range(16):
#     granularity.append(i+1)

# conv2v1 = [15.463968,8.451616,8.801440,7.408416,7.452768,8.842400,9.468416,9.347200,9.783360,9.911648,10.555584,11.114432,12.073824,12.196032,13.334175,13.436640]
# conv2v2 = [5.158304,4.839104,5.699584,5.989376,7.366656,7.352320,7.813120,8.649728,9.426944,10.178560,10.377216,10.376192,11.112448,11.885312,12.695264,13.508608]
# conv2v3 = [6.384352,5.938176,6.292480,6.624224,8.141792,8.575744,9.562112,10.563584,11.765760,12.726272,13.808640,14.808064,15.852544,17.446913,16.989183,17.636353]
# conv2v4 = [6.312256,6.238112,6.348800,5.960672,5.903360,6.419232,6.227968,6.959104,6.354752,5.796768,6.356640,6.888448,7.476224,8.039424,8.587200,9.116416]

# plt.title('layer conv_2 perfomance for different version across granularities')
# plt.ylabel('execution time in ms')
# plt.xlabel('granularity')
# plt.plot(granularity,conv2v1,color='r',label='p1:local mask')
# plt.plot(granularity,conv2v2,color='b',label='p1:shared memory mask')
# plt.plot(granularity,conv2v3,color='y',label='p1:shared memory mask and ifm planes')
# plt.plot(granularity,conv2v4,color='m',label='p2:shared memory mask planes and ifm planes')
# plt.legend(loc='upper left')
# plt.show()




# granularity = []
# for i in range(13):
#     granularity.append(i+1)
# # conv3v1 = [18.079136,10.252384,12.379296,14.795712,17.242208,19.794624,21.863424,24.739361,27.516928,30.415392,33.230946,36.057758,38.880833]
# conv3v1 = [10.696192,4.126016,5.329696,6.428480,7.516832,8.513216,9.613760,10.844192,12.071936,12.711456,13.605152,14.725280,15.858688]
# conv3v2 = [1.897248,2.975072,4.049312,5.104544,6.067328,6.950432,7.959872,8.994112,10.096960,11.184864,12.276160,13.020448,13.013024]
# conv3v3 = [2.073024,3.617760,5.198880,6.765728,8.429760,9.627136,11.570624,12.882048,14.182112,15.526656,16.835936,17.984961,19.459969]
# conv3v4 = [1.568576,1.569024,1.784640,1.764512,2.240544,2.711552,2.971104,3.213280,3.628128,4.149344,4.496672,4.766720,5.192224]

# plt.title('layer conv_3 perfomance for different version across granularities')
# plt.ylabel('execution time in ms')
# plt.xlabel('granularity')

# plt.plot(granularity,conv3v1,color='r',label='p1:local mask')
# plt.plot(granularity,conv3v2,color='b',label='p1:shared memory mask')
# plt.plot(granularity,conv3v3,color='y',label='p1:shared memory mask and ifm planes')
# plt.plot(granularity,conv3v4,color='m',label='p2:shared memory mask planes and ifm planes')
# plt.legend(loc='upper left')
# plt.show()

# granularity = []
# for i in range(13):
#     granularity.append(i+1)

# conv4v1 = [15.300032,12.374016,14.629792,11.089920,11.850752,12.774080,13.194368,14.758560,16.309248,17.848320,19.400703,19.943424,21.377024]
# conv4v2 = [2.569408,4.024384,5.758272,7.385088,9.208256,10.785856,12.257984,13.625856,15.393632,16.695841,17.086081,18.571615,19.902784]
# conv4v3 = [3.115264,5.402624,7.792128,10.161792,12.643264,14.434720,17.383392,19.321119,21.297344,23.295200,25.255009,27.260609,31.948671]
# conv4v4 = [3.722912,3.329408,4.274016,2.654400,3.382688,4.104064,4.460064,4.845088,5.546560,6.273248,6.788960,7.158656,7.800416]


# plt.title('layer conv_4 perfomance for different version across granularities')
# plt.ylabel('execution time in ms')
# plt.xlabel('granularity')

# plt.plot(granularity,conv4v1,color='r',label='p1:local mask')
# plt.plot(granularity,conv4v2,color='b',label='p1:shared memory mask')
# plt.plot(granularity,conv4v3,color='y',label='p1:shared memory mask and ifm planes')
# plt.plot(granularity,conv4v4,color='m',label='p2:shared memory mask planes and ifm planes')
# plt.legend(loc='upper left')
# plt.show()


granularity = []
for i in range(13):
    granularity.append(i+1)

# conv5v1 = [11.218016,6.924288,7.943168,8.721184,9.887744,11.127808,12.170240,13.620224,15.019008,16.479233,17.968992,19.196928,19.885056]
conv5v1 = [12.278400,5.546272,7.268096,8.838656,10.514112,12.125792,13.852512,15.673120,16.698879,18.239103,19.955008,21.868608,23.336161]
conv5v2 = [2.426144,3.926240,5.640096,7.189952,9.038592,10.620864,12.023968,13.427552,15.186880,16.771711,18.327265,19.916800,21.360737]
# conv5v2 = [4.686464,8.813568,12.923904,16.519169,20.538368,24.525824,28.085152,30.620672,32.301056,35.829762,39.195553,42.526657,42.177505]
# conv5v3 = [5.131808,8.881152,12.831712,16.657408,20.806656,23.915520,28.525568,31.757120,34.997250,35.919872,37.538815,40.479744,46.616577]
conv5v3 = [2.391648,4.047616,5.836288,7.592896,9.455744,10.813344,12.981920,14.405952,15.437856,16.172001,17.484768,18.878208,22.037664]
conv5v4 = [2.623328,3.047232,2.228160,2.623808,3.173120,3.855200,4.400864,4.812704,5.447392,6.160128,6.684384,7.078176,7.717152]

plt.title('layer conv_5 perfomance for different version across granularities')
plt.ylabel('execution time in ms')
plt.xlabel('granularity')

plt.plot(granularity,conv5v1,color='r',label='p1:local mask')
plt.plot(granularity,conv5v2,color='b',label='p1:shared memory mask')
plt.plot(granularity,conv5v3,color='y',label='p1:shared memory mask and ifm planes')
plt.plot(granularity,conv5v4,color='m',label='p2:shared memory mask planes and ifm planes')
plt.legend(loc='upper left')
plt.show()
