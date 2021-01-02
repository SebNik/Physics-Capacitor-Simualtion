import numpy as np
import pandas as pd
from Plate_Capacitor import Plate_Capacitor

if __name__ == "__main__":
    # setting the path for the big sims
    path = 'D:\\Python\\Programme\\Physics-Capacitor-Simualtion\\resources\\Test_Simulationen_4.xlsx'
    # opening the csv file
    df = pd.read_excel(path)
    print(df)
    # iterating through the df
    for index, row in df.iterrows():
        print('--------------------------- ' + str(index) + ' ---------------------------')
        p1 = [float(row['p1_x']), float(row['p1_y'])]
        p2 = [float(row['p2_x']), float(row['p2_y'])]
        cap = Plate_Capacitor(n_neg=int(row['n_neg']), n_pos=int(row['n_pos']), p1=p1, p2=p2,
                              plane_z_pos=[float(row['plane_z_pos'])], plane_z_neg=[float(row['plane_z_neg'])],
                              random=bool(row['random']), name=row['name'])
        if bool(row['sim']):
            cap.sim(t=0.0000001)
        if bool(row['field_lines']):
            x = [0.015]
            cap.plot_field_lines_integral_calculation_flatten(num_field_lines=50, x_plane=x, nbins=30,
                                                              delta_m=0.000004 * 25)
