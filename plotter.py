import matplotlib.pyplot as plt

granularity = []
for i in range(16):
    granularity.append(i+1)
conv2v1 = [34.058304,25.721504,27.513599,23.327648,23.722464,26.244352,28.87023,28.308384,27.756256,29.259487,30.058687,30.237663,31.234560,32.547775,34.188641,35.854015]
conv2v2 = [11.141312,11.777248,13.243328,15.066624,18.466688,21.252384,23.835360,26.566463,29.923967,32.405792,35.239902,38.399231,41.394016,44.681984,47.377602,50.455170]
conv2v3 = [10.101088,15.345600,21.475519,27.104671,33.054241,39.236546,46.087425,50.973343,59.617023,63.858528,68.432800,73.497505,78.343262,90.242973,93.021378,96.295776]
conv2v4 = [22.965088,13.393248,9.402784,12.712192,12.733280,11.957248,13.396736,13.643520,9.065472,10.836320,10.673696,11.817280,11.316608,13.166016,13.952384,14.444800]

plt.title('conv_2 layer perfomance for different version across granularities')
plt.ylabel('execution time in ms')
plt.xlabel('granularity')

plt.plot(granularity,conv2v1,color='r',label='p1:local mask')
plt.plot(granularity,conv2v2,color='b',label='p1:shared memory mask')
plt.plot(granularity,conv2v3,color='y',label='p1:shared memory mask and ifm planes')
plt.plot(granularity,conv2v4,color='m',label='p2:shared memory mask planes and ifm planes')
plt.legend(loc='upper left')
plt.show()
