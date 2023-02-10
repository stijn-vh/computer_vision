import online_phase as Online
import offline_phase as Offline

if __name__ == '__main__':
    params = Offline.offline_phase.execute_offline_phase()
    
    Online.online_phase.execute_online_phase(params)
