import csv


def inputClassification():

    path = 'testClassification/test1.csv'
    with open(path, 'w') as f:
        w = csv.writer(f, delimiter=',')
        w.writerow(['SalePrice', 'YearBuilt', 'YrSold', 'MonthSold', 'Size(sqf)', 'Floor',
                             'N_Parkinglot(Ground)', 'N_Parkinglot(Basement)', 'TimeToBusStop', 'TimeToSubway',
                             'N_APT', 'N_manager', 'N_elevators', 'N_FacilitiesNearBy(PublicOffice)',
                             'N_FacilitiesNearBy(Hospital)', 'N_FacilitiesNearBy(Dpartmentstore)',
                             'N_FacilitiesNearBy(Mall)', 'N_FacilitiesNearBy(ETC)', 'N_FacilitiesNearBy(Park)',
                             'N_SchoolNearBy(Elementary)', 'N_SchoolNearBy(Middle)', 'N_SchoolNearBy(High)',
                             'N_SchoolNearBy(University)', 'N_FacilitiesInApt', 'N_FacilitiesNearBy(Total)',
                             'N_SchoolNearBy(Total)'])

        flag = True

        while flag:
                price = float(input_validation("Insert value for SalePrice: "))
                ybuilt = int(input_validation("Insert value for YearBuilt: "))
                ysold = int(input_validation("Insert value for YrSold: "))
                msold = int(input_validation_month("Insert value for MonthSold: "))
                size = float(input_validation("Insert value for Size(sqf): "))
                floor = int(input_validation("Insert value for Floor: "))
                parking1 = int(input_validation("Insert value for N_Parkinglot(Ground): "))
                parking2 = int(input_validation("Insert value for N_Parkinglot(Basement): "))
                bus = float(input_validation("Insert value for TimeToBusStop: "))
                subway = float(input_validation("Insert value for TimeToSubway: "))
                apt = int(input_validation("Insert value for N_APT: "))
                manager = int(input_validation("Insert value for N_manager: "))
                elevator = int(input_validation("Insert value for N_elevators: "))
                office = int(input_validation("Insert value for N_FacilitiesNearBy(PublicOffice): "))
                hospital = int(input_validation("Insert value for N_FacilitiesNearBy(Hospital): "))
                dstore = int(input_validation("Insert value for N_FacilitiesNearBy(Dpartmentstore): "))
                mall = int(input_validation("Insert value for N_FacilitiesNearBy(Mall): "))
                etc = int(input_validation("Insert value for N_FacilitiesNearBy(ETC): "))
                park = int(input_validation("Insert value for N_FacilitiesNearBy(Park): "))
                elementary = int(input_validation("Insert value for N_SchoolNearBy(Elementary): "))
                middle = int(input_validation("Insert value for N_SchoolNearBy(Middle): "))
                high = int(input_validation("Insert value for N_SchoolNearBy(High): "))
                university = int(input_validation("Insert value for N_SchoolNearBy(University): "))
                fapt = int(input_validation("Insert value for N_FacilitiesInApt: "))
                total1 = int(input_validation("Insert value for N_FacilitiesNearBy(Total): "))
                total2 = int(input_validation("Insert value for N_SchoolNearBy(Total): "))
                w.writerow([price, ybuilt, ysold, msold, size, floor, parking1, parking2, bus, subway, apt,
                                  manager, elevator, office, hospital, dstore, mall, etc, park, elementary, middle,
                                  high, university, fapt, total1, total2])
                flag = False

    f.close()

    return path


def inputPrediction():
    path = 'testRegression/test1.csv'
    with open(path, 'w') as f:
        w = csv.writer(f, delimiter=',')
        w.writerow(['YearBuilt', 'YrSold', 'MonthSold', 'Size(sqf)', 'Floor',
                    'N_Parkinglot(Ground)', 'N_Parkinglot(Basement)', 'TimeToBusStop', 'TimeToSubway',
                    'N_APT', 'N_manager', 'N_elevators', 'N_FacilitiesNearBy(PublicOffice)',
                    'N_FacilitiesNearBy(Hospital)', 'N_FacilitiesNearBy(Dpartmentstore)',
                    'N_FacilitiesNearBy(Mall)', 'N_FacilitiesNearBy(ETC)', 'N_FacilitiesNearBy(Park)',
                    'N_SchoolNearBy(Elementary)', 'N_SchoolNearBy(Middle)', 'N_SchoolNearBy(High)',
                    'N_SchoolNearBy(University)', 'N_FacilitiesInApt', 'N_FacilitiesNearBy(Total)',
                    'N_SchoolNearBy(Total)'])

        flag = True

        while flag:
                ybuilt = int(input_validation("Insert value for YearBuilt: "))
                ysold = int(input_validation("Insert value for YrSold: "))
                msold = int(input_validation_month("Insert value for MonthSold: "))
                size = float(input_validation("Insert value for Size(sqf): "))
                floor = int(input_validation("Insert value for Floor: "))
                parking1 = int(input_validation("Insert value for N_Parkinglot(Ground): "))
                parking2 = int(input_validation("Insert value for N_Parkinglot(Basement): "))
                bus = float(input_validation("Insert value for TimeToBusStop: "))
                subway = float(input_validation("Insert value for TimeToSubway: "))
                apt = int(input_validation("Insert value for N_APT: "))
                manager = int(input_validation("Insert value for N_manager: "))
                elevator = int(input_validation("Insert value for N_elevators: "))
                office = int(input_validation("Insert value for N_FacilitiesNearBy(PublicOffice): "))
                hospital = int(input_validation("Insert value for N_FacilitiesNearBy(Hospital): "))
                dstore = int(input_validation("Insert value for N_FacilitiesNearBy(Dpartmentstore): "))
                mall = int(input_validation("Insert value for N_FacilitiesNearBy(Mall): "))
                etc = int(input_validation("Insert value for N_FacilitiesNearBy(ETC): "))
                park = int(input_validation("Insert value for N_FacilitiesNearBy(Park): "))
                elementary = int(input_validation("Insert value for N_SchoolNearBy(Elementary): "))
                middle = int(input_validation("Insert value for N_SchoolNearBy(Middle): "))
                high = int(input_validation("Insert value for N_SchoolNearBy(High): "))
                university = int(input_validation("Insert value for N_SchoolNearBy(University): "))
                fapt = int(input_validation("Insert value for N_FacilitiesInApt: "))
                total1 = int(input_validation("Insert value for N_FacilitiesNearBy(Total): "))
                total2 = int(input_validation("Insert value for N_SchoolNearBy(Total): "))
                w.writerow([ybuilt, ysold, msold, size, floor, parking1, parking2, bus, subway, apt,
                            manager, elevator, office, hospital, dstore, mall, etc, park, elementary, middle,
                            high, university, fapt, total1, total2])
                flag = False

    f.close()

    return path


def input_validation_month(prompt):

    flag = True
    while flag:
        answer = input(prompt)
        try:
            month = int(answer)
            if month <= 12:
                flag = False
                return month
            else:
                print('Please insert value between 1 and 12!')
        except ValueError:
            print('Please insert only numerical value!')


def input_validation(prompt):

    flag = True
    while flag:
        answer = input(prompt)
        try:
            value = float(answer)
            if value.is_integer():
                flag = False
                return value
            else:
                print('Please insert numerical value!')
        except ValueError:
            print('Please insert only numerical value!')
