import online_phase as Online
import offline_phase as Offline

if __name__ == '__main__':
    params = Offline.execute_offline_phase()
    print('Offline executed')
    Online.execute_online_phase(params)
 