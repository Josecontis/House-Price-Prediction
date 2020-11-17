import pandas as pd
import numpy as np


df = pd.read_csv('../extraction/datasetID.csv')

id = np.array(df['Id'])
park = np.array(df['N_FacilitiesNearBy(Park)'])
hospital = np.array(df['N_FacilitiesNearBy(Hospital)'])
office = np.array(df['N_FacilitiesNearBy(PublicOffice)'])
floor = np.array(df['Floor'])
elevator = np.array(df['N_elevators'])
university = np.array(df['N_SchoolNearBy(University)'])
mall = np.array(df['N_FacilitiesNearBy(Mall)'])
etc = np.array(df['N_FacilitiesNearBy(ETC)'])
bus = np.array(df['TimeToBusStop'])
subway = np.array(df['TimeToSubway'])
elementary = np.array(df['N_SchoolNearBy(Elementary)'])
middle = np.array(df['N_SchoolNearBy(Middle)'])
high = np.array(df['N_SchoolNearBy(High)'])
parking1 = np.array(df['N_Parkinglot(Ground)'])
parking2 = np.array(df['N_Parkinglot(Basement)'])
size = np.array(df['Size(sqf)'])

for c in range(len(id)):
    if (park.item(c) > 0) & (office.item(c) > 2) & (floor.item(c) == 1):
        df.loc[:, 'AptType'][c] = "Old"
    elif ((2 >= office.item(c) >= 3) & (mall.item(c) > 0) & (900 >= size.item(c) >= 1400)
          & (1 >= parking1.item(c) >= 4) or (1 >= parking2.item(c) >= 4)
          & (elementary.item(c) > 1) or (middle.item(c) > 1) or (high.item(c) > 1)):
        df.loc[:, 'AptType'][c] = "Family"
    elif ((park.item(c) > 0) & (etc.item(c) > 0) & (size.item(c) < 600) & (university.item(c) >= 1)
          or (2 >= bus.item(c) >= 7.5) or (0 >= subway.item(c) >= 12.5)):
        df.loc[:, 'AptType'][c] = "University Student"
    elif ((2 >= office.item(c) >= 3) & (mall.item(c) > 0) & (700 >= size.item(c) >= 1100)
          & (parking1.item(c) > 0) or (parking2.item(c) > 0)):
        df.loc[:, 'AptType'][c] = "Married Couple"
    elif (office.item(c) >= 1) & (park.item(c) > 0):
        df.loc[:, 'AptType'][c] = "Single"

# write out the csv
df.to_csv('../extraction/datasetAptType.csv', index=False, float_format='%.2f')

