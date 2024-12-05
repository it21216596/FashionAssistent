

import numpy as np
import sys
import os
import utils

DATA_DIR = "data"
# read control  points(CP) from text file
def convert_cp():
    
  f = open(os.path.join(DATA_DIR, 'customBodyPoints.txt'), "r")

  tmplist = []
  cp = []
  for line in f:
    if '#' in line:
      if len(tmplist) != 0:
        cp.append(tmplist)
        tmplist = []
    elif len(line.split()) == 1:
      continue
    else:
      tmplist.append(list(map(float, line.strip().split())))
  cp.append(tmplist)


  return cp


# calculate measure data from given vertex by control points
def calc_measure(cp, vertex,height):#, facet):
  measure_list = []
  
  for measure in cp:


    length = 0.0
    p2 = vertex[int(measure[0][1]), :]

    for i in range(0, len(measure)):#1
      p1 = p2
      if measure[i][0] == 1:
        p2 = vertex[int(measure[i][1]), :]  
        
      elif measure[i][0] == 2:
        p2 = vertex[int(measure[i][1]), :] * measure[i][3] + \
        vertex[int(measure[i][2]), :] * measure[i][4]
#        print("if 2 Measurement",int(measure[i][1]))
        
      else:
        p2 = vertex[int(measure[i][1]), :] * measure[i][4] + \
          vertex[int(measure[i][2]), :] * measure[i][5] + \
          vertex[int(measure[i][3]), :] * measure[i][6]
      length += np.sqrt(np.sum((p1 - p2)**2.0))

    measure_list.append(length * 100)# * 1000
  
  measure_list = float(height)*(measure_list/measure_list[0])

  measure_list[8] = measure_list[8] * 0.36
  measure_list[3] = measure_list[3] * 0.6927

  return np.array(measure_list).reshape(utils.M_NUM, 1)



def extract_measurements(height, vertices):
  genders = ["male"]#, "male"]
  measure = []
  for gender in genders:
   
    cp = convert_cp()


    measure = calc_measure(cp, vertices, height)


    #give body measurements one by one
    for i in range(0, utils.M_NUM):
      print("%s: %f" % (utils.M_STR[i], measure[i]))
    
    
    
    face_path = './src/tf_smpl/smpl_faces.npy'
    faces = np.load(face_path)
    obj_mesh_name = 'test.obj'
    with open(obj_mesh_name, 'w') as fp:
        for v in vertices:
            fp.write( 'v %f %f %f\n' % ( v[0], v[1], v[2]) )
        for f in faces: # Faces are 1-based, not 0-based in obj files
            fp.write( 'f %d %d %d\n' %  (f[0] + 1, f[1] + 1, f[2] + 1) )

        
    print("Model Saved...")
