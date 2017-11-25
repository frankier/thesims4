import random
import simpy
import pandas
from collections import deque


RANDOM_SEED = 42
JOB_DURATION = 30
SIM_TIME = 1000

LIGHT_PATIENT = 1
SEVERE_PATIENT = 2

L_IAT = 20
S_IAT = 30

T_PLP = 30
T_PSP = 50
P = 3

T_OLP = 15
T_OSP = 25

T_RLP = 30
T_RSP = 50
R = 3


def randexp(x):
    return random.expovariate(1 / x)


class PredEvtWaitQueue:
    """
    Takes a function returning a predicate and an function returning an event.

    When someone queues, will return an event that either is succeeded
    instantly if the predicat is true otherwise will succeed when an event is
    triggered is true in FIFO order.
    """

    def __init__(self, env, pred, get_evt):
        self.evts = deque()
        self.env = env
        self.pred = pred

        def loop():
            while True:
                if self.evts:
                    evt = self.evts.popleft()
                    evt.succeed()
                # Put self to back of queue so new evt can be created
                yield self.env.timeout(0)
                yield get_evt()
                print('evt succeed')
        env.process(loop())

    def queue(self):
        evt = self.env.event()
        if self.pred():
            evt.succeed()
        else:
            self.evts.append(evt)
        return evt


class Patient:
    def __init__(self, env):
        self.env = env
        self.creation_time = self.env.now
        self.prep_time = randexp(self.mean_prep_time)
        self.surgery_time = randexp(self.mean_surgery_time)
        self.recovery_time = randexp(self.mean_recovery_time)
        self.process = env.process(self.run())

    def run(self):
        #print("I'm a patient going to prep")
        self.done = env.event()
        pipeline['prep'].arrival.succeed(self)
        #print("I've gone to prep now")
        yield self.done
        #print("I've been prepped")
        self.done = env.event()
        pipeline['surg'].arrival.succeed(self)
        yield self.done
        self.done = env.event()
        pipeline['recov'].arrival.succeed(self)
        yield self.done
        print("I'm done")


class LightPatient(Patient):
    mean_prep_time = T_PLP
    mean_surgery_time = T_OLP
    mean_recovery_time = T_RLP


class SeverePatient(Patient):
    mean_prep_time = T_PSP
    mean_surgery_time = T_OSP
    mean_recovery_time = T_RSP


class PatientGenerator:
    def __init__(self, env):
        self.env = env
        self.process = env.process(self.run())

    def run(self):
        while True:
            yield self.env.timeout(randexp(self.iat))
            patient = self.patient_cls(self.env)
            patients.append(patient)


class LightPatientGenerator(PatientGenerator):
    patient_type = LIGHT_PATIENT
    patient_cls = LightPatient
    iat = L_IAT


class SeverePatientGenerator(PatientGenerator):
    patient_type = SEVERE_PATIENT
    patient_cls = SeverePatient
    iat = S_IAT


class RoomContainer:
    def __init__(self, env):
        self.env = env
        self.arrival = env.event()
        self.room_freed = env.event()
        self.free_queue = deque()
        self.patient_queue = deque()
        self.mk_initial_rooms()
        self.rooms = list(self.free_queue)
        self.process = env.process(self.run())

    def run(self):
        while True:
            evt = yield self.arrival | self.room_freed
            #print("I'm a prep room that just got an event")
            if self.room_freed in evt:
                room = evt[self.room_freed]
                self.room_freed = env.event()
                if self.patient_queue:
                    patient = self.patient_queue.popleft()
                    room.arrival.succeed(patient)
                else:
                    self.free_queue.append(room)
            if self.arrival in evt:
                patient = evt[self.arrival]
                self.arrival = env.event()
                #print("A patient just arrived")
                if self.free_queue:
                    room = self.free_queue.popleft()
                    room.arrival.succeed(patient)
                else:
                    self.patient_queue.append(patient)
                #self.patient_queue.append(patient)

    @property
    def is_utilised(self):
        return sum(r.is_utilised for r in self.rooms)

    @property
    def is_occupied(self):
        return sum(r.is_occupied for r in self.rooms)


class PreparationRooms(RoomContainer):
    def mk_initial_rooms(self):
        for i in range(P):
            self.free_queue.append(PreparationRoom(self))


class Room:
    def __init__(self, container):
        self.container = container
        self.env = container.env
        self.process = env.process(self.run())
        self.is_utilised = False
        self.is_occupied = False

    def extra_block(self, patient):
        return self.env.timeout(0)

    def run(self):
        #print("Hi there I'm a preperation room")
        while True:
            self.arrival = env.event()
            patient = yield self.arrival
            self.is_utilised = True
            self.is_occupied = True
            #print("Someone's arrived in prep")
            yield env.timeout(self.get_occupation_time(patient))
            self.is_utilised = False
            yield self.extra_block(patient)
            self.is_occupied = False
            print("About to succeed", self, patient)
            patient.done.succeed()
            print("About to free", self.env.now, self, self.container)
            self.container.room_freed.succeed(self)


class PreparationRoom(Room):
    def get_occupation_time(self, patient):
        return patient.prep_time

    def extra_block(self, patient):
        return wait_until_surgery_free.queue()


class Surgery:
    def __init__(self, env):
        self.env = env
        self.patients_processed = 0
        self.process = env.process(self.run())
        #self.utilization_time = 0
        self.is_utilised = False
        self.is_occupied = False

    def run(self):
        while True:
            self.freed = env.event()
            self.arrival = env.event()

            yield self.arrival
            patient = self.arrival.value
            start = self.env.now
            self.is_utilised = True
            self.is_occupied = True
            yield self.env.timeout(patient.surgery_time)
            #self.utilization_time += self.env.now - start
            self.is_utilised = False
            yield wait_until_recovery_free.queue()
            print('About to succeed', self, patient)
            patient.done.succeed()
            self.freed.succeed()
            self.is_occupied = False

            # Patient is done.
            self.patients_processed += 1


class RecoveryRooms(RoomContainer):
    def mk_initial_rooms(self):
        for i in range(R):
            self.free_queue.append(RecoveryRoom(self))

    def is_available(self):
        return bool(self.free_queue)


class RecoveryRoom(Room):
    def get_occupation_time(self, patient):
        return patient.recovery_time


# Setup and start the simulation
print('Surgery')
random.seed(RANDOM_SEED)  # This helps reproducing the results

# Create an environment and start the setup process
env = simpy.Environment()
#machines = [Machine(env, 'Machine %d' % i)
            #for i in range(NUM_MACHINES)]

pipeline = {
    'prep': PreparationRooms(env),
    'surg': Surgery(env),
    'recov': RecoveryRooms(env),
}
lpg = LightPatientGenerator(env)
spg = SeverePatientGenerator(env)
patients = []

wait_until_surgery_free = PredEvtWaitQueue(
    env,
    lambda: not pipeline['surg'].is_occupied,
    lambda: pipeline['surg'].freed)

wait_until_recovery_free = PredEvtWaitQueue(
    env,
    lambda: pipeline['recov'].is_available(),
    lambda: pipeline['recov'].room_freed)


# Monitoring
class Monitor():
    def __init__(self, env):
        self.env = env
        self.columns = ['time'] + list(probes.keys())
        # XXX: should maybe be int instead but this will work well enough
        self.data = pandas.DataFrame(columns=self.columns, dtype=float)
        self.process = env.process(self.run())

    def run(self):
        i = 0
        while True:
            row = [self.env.now]
            for probe_name, probe_fn in probes.items():
                row.append(probe_fn())
            self.data.loc[i] = row
            yield self.env.timeout(1)
            i += 1


def probe(fn):
    probes[fn.__name__] = fn
    return fn


probes = {}


def add_utilisation_probes(stage):
    def utilisation():
        return int(pipeline[stage].is_utilised)
    probes['{}_utilisation'.format(stage)] = utilisation

    def occupation():
        return int(pipeline[stage].is_occupied)
    probes['{}_occupation'.format(stage)] = occupation

    def backed_up():
        return pipeline[stage].is_occupied - pipeline[stage].is_utilised
    probes['{}_backed_up'.format(stage)] = backed_up


add_utilisation_probes('prep')
add_utilisation_probes('surg')
add_utilisation_probes('recov')

@probe
def prep_queue():
    return len(pipeline['prep'].patient_queue)

@probe
def prep_free():
    return len(pipeline['prep'].free_queue)

@probe
def recov_free():
    return len(pipeline['recov'].free_queue)

monitor = Monitor(env)

# Execute!
env.run(until=SIM_TIME)

# Analyis/results
print('Hospital results after {} time steps'.format(SIM_TIME))
#print('Was utilized for {} time steps'.format(
    #pipeline['surg'].utilization_time))
print('Processed {} patients'.format(
    pipeline['surg'].patients_processed))

import matplotlib.pyplot as plt
#plt.figure()
monitor.data.hist(figsize=(10,9), column=list(probes.keys()))
plt.tight_layout()
plt.savefig('distr.pdf')
print('Distributions saved to distr.pdf')

print('Mean preparation queue length', monitor.data['prep_queue'].mean())
print('Mean surgery utilisation', monitor.data['surg_utilisation'].mean())
