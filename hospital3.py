import sys
import pickle
import random
import simpy
import pandas
import numpy
from collections import deque
import matplotlib.pyplot as plt
from simpy.events import Event


RANDOM_SEED = 42
JOB_DURATION = 30


def randexp(mean, lower, upper):
    return random.expovariate(1 / mean)


def randunif(mean, lower, upper):
    return random.uniform(lower, upper)


def call_sexp(sexp):
    return sexp[0](*sexp[1:])


DEFAULT_PG_PARAMS = {
    'iat': (randexp, 25, -1, -1),
    'prep_time': (randexp, 40, -1, -1),
    'surgery_time': (randexp, 20, -1, -1),
    'recovery_time': (randexp, 40, -1, -1),
}


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
        env.process(loop())

    def queue(self):
        evt = self.env.event()
        if self.pred():
            evt.succeed()
        else:
            self.evts.append(evt)
        return evt


class Patient:
    def __init__(self, env, pipeline, params):
        self.env = env
        self.pipeline = pipeline
        self.creation_time = self.env.now
        self.prep_time = call_sexp(params['prep_time'])
        self.surgery_time = call_sexp(params['surgery_time'])
        self.recovery_time = call_sexp(params['recovery_time'])
        self.process = env.process(self.run())

    def run(self):
        self.done = self.env.event()
        self.pipeline.prep.arrival.succeed(self)
        yield self.done
        self.done = self.env.event()
        self.pipeline.surg.arrival.succeed(self)
        yield self.done
        self.done = self.env.event()
        self.pipeline.recov.arrival.succeed(self)
        yield self.done


class PatientGenerator:
    def __init__(self, env, pipeline, params):
        self.env = env
        self.pipeline = pipeline
        self.process = env.process(self.run())
        self.params = params.copy()
        self.iat = self.params.pop('iat')

    def run(self):
        while True:
            yield self.env.timeout(call_sexp(self.iat))
            Patient(self.env, self.pipeline, self.params)


class RoomContainer:
    def __init__(self, env, pipeline, num_rooms):
        self.env = env
        self.pipeline = pipeline
        self.arrival = env.event()
        self.room_freed = env.event()
        self.free_queue = deque()
        self.patient_queue = deque()
        self.num_rooms = num_rooms
        self.mk_initial_rooms()
        self.rooms = list(self.free_queue)
        self.process = env.process(self.run())

    def mk_initial_rooms(self):
        for i in range(self.num_rooms):
            self.free_queue.append(self.room_type(self))

    def is_available(self):
        #print(self, 'is available', bool(self.free_queue))
        return bool(self.free_queue)

    def run(self):
        while True:
            evt = yield self.arrival | self.room_freed
            if self.room_freed in evt:
                room = evt[self.room_freed]
                self.room_freed = self.env.event()
                if self.patient_queue:
                    patient = self.patient_queue.popleft()
                    room.arrival.succeed(patient)
                else:
                    self.free_queue.append(room)
            if self.arrival in evt:
                patient = evt[self.arrival]
                self.arrival = self.env.event()
                if self.free_queue:
                    room = self.free_queue.popleft()
                    room.arrival.succeed(patient)
                else:
                    self.patient_queue.append(patient)

    @property
    def is_utilised(self):
        return sum(r.is_utilised for r in self.rooms)

    @property
    def is_occupied(self):
        return sum(r.is_occupied for r in self.rooms)


class Room:
    def __init__(self, container):
        self.container = container
        self.env = container.env
        self.process = self.env.process(self.run())
        self.is_utilised = False
        self.is_occupied = False

    def extra_block(self, patient):
        return self.env.timeout(0)

    def run(self):
        while True:
            self.arrival = self.env.event()
            patient = yield self.arrival
            self.is_utilised = True
            self.is_occupied = True
            yield self.env.timeout(self.get_occupation_time(patient))
            self.is_utilised = False
            yield self.extra_block(patient)
            self.is_occupied = False
            patient.done.succeed()
            self.container.room_freed.succeed(self)


class PreparationRoom(Room):
    def get_occupation_time(self, patient):
        return patient.prep_time

    def extra_block(self, patient):
        return self.container.pipeline.wait_until_surgery_free.queue()


class PreparationRooms(RoomContainer):
    room_type = PreparationRoom


class Surgery:
    def __init__(self, env, pipeline):
        self.env = env
        self.pipeline = pipeline
        self.patients_processed = 0
        self.process = env.process(self.run())
        #self.utilization_time = 0
        self.is_utilised = False
        self.is_occupied = False

    def run(self):
        while True:
            self.freed = self.env.event()
            self.arrival = self.env.event()

            yield self.arrival
            patient = self.arrival.value
            start = self.env.now
            self.is_occupied = True
            self.is_utilised = True
            yield self.env.timeout(patient.surgery_time)
            #self.utilization_time += self.env.now - start
            self.is_utilised = False
            yield self.pipeline.wait_until_recovery_free.queue()
            patient.done.succeed()
            self.freed.succeed()
            self.is_occupied = False

            # Patient is done.
            self.patients_processed += 1


class RecoveryRoom(Room):
    def get_occupation_time(self, patient):
        return patient.recovery_time


class RecoveryRooms(RoomContainer):
    room_type = RecoveryRoom


class Pipeline:
    def __init__(self, env, preparation_rooms, recovery_rooms, patient_generators):
        self.prep = PreparationRooms(env, self, preparation_rooms)
        self.surg = Surgery(env, self)
        self.recov = RecoveryRooms(env, self, recovery_rooms)
        self.patient_generators = []
        for patient_generator in patient_generators:
            self.patient_generators.append(PatientGenerator(env, self, patient_generator))

        self.wait_until_surgery_free = PredEvtWaitQueue(
            env,
            lambda: not self.surg.is_occupied,
            lambda: self.surg.freed)

        self.wait_until_recovery_free = PredEvtWaitQueue(
            env,
            lambda: self.recov.is_available(),
            lambda: self.recov.room_freed)


# Setup and start the simulation
random.seed(RANDOM_SEED)  # This helps reproducing the results


# Monitoring
class Monitor():
    def __init__(self, env, probes):
        self.env = env
        self.probes = probes
        self.columns = ['time'] + list(probes.keys())
        # XXX: should maybe be int instead but this will work well enough
        self.data = pandas.DataFrame(columns=self.columns, dtype=float)
        self.i = 0
        self.schedule(0)

    def schedule(self, delay):
        evt = Event(self.env)
        evt.callbacks.append(self.observe)
        evt._ok = True
        self.env.schedule(evt, priority=2, delay=delay)

    def observe(self, _evt):
        # Need to be last thing scheduled in a tick, so as to avoid
        # observing an indeterminate state.
        if self.i % 1000 == 0:
            print('Iter', self.i)
        row = [self.env.now]
        for probe_name, probe_fn in self.probes.items():
            row.append(probe_fn())
        self.data.loc[self.i] = row
        self.i += 1
        self.schedule(1)


def run_experiment(time=1000, preparation_rooms=3, recovery_rooms=3, patient_generators=(DEFAULT_PG_PARAMS,)):
    # Create an environment and start the setup process
    env = simpy.Environment()

    pipeline = Pipeline(env, preparation_rooms, recovery_rooms, patient_generators)
    monitor = mk_monitor(env, pipeline)

    # Execute!
    env.run(until=time)

    return pipeline, monitor


def mk_monitor(env, pipeline):
    def probe(fn):
        probes[fn.__name__] = fn
        return fn

    probes = {}

    def add_utilisation_probes(stage):
        def utilisation():
            return int(getattr(pipeline, stage).is_utilised)
        probes['{}_utilisation'.format(stage)] = utilisation

        def occupation():
            return int(getattr(pipeline, stage).is_occupied)
        probes['{}_occupation'.format(stage)] = occupation

        def backed_up():
            return getattr(pipeline, stage).is_occupied - getattr(pipeline, stage).is_utilised
        probes['{}_backed_up'.format(stage)] = backed_up

    add_utilisation_probes('prep')
    add_utilisation_probes('surg')
    add_utilisation_probes('recov')

    @probe
    def prep_queue():
        return len(pipeline.prep.patient_queue)

    @probe
    def prep_free():
        return len(pipeline.prep.free_queue)

    @probe
    def recov_free():
        return len(pipeline.recov.free_queue)

    @probe
    def recov_full():
        return len(pipeline.recov.free_queue) == 0

    return Monitor(env, probes)


#def report_results(pipeline, monitor):
#    # Analyis/results
#    print('Processed {} patients'.format(
#        pipeline.surg.patients_processed))
#
#    monitor.data.hist(figsize=(10,9), column=list(monitor.probes.keys()))
#    plt.tight_layout()
#    plt.savefig('distr.pdf')
#    print('Distributions saved to distr.pdf')
#
#    print('Mean preparation queue length', monitor.data['prep_queue'].mean())
#    print('Mean surgery utilisation', monitor.data['surg_utilisation'].mean())
#
#
#def mk_sampler(probes, slice):
#    def inner(pipeline, monitor):
#        return list(zip(
#            *(monitor.data[probe].iloc[slice] for probe in probes)))
#
#    return inner
#
#
#def tasks(out_f, variables, reseed=False):
#    # Observe the length of a queue before preparation room, the length of idle
#    # queue for preparation rooms and the probability of operation being in
#    # waiting state (blocked). You can do this by sampling the variables with
#    # regular rate. Compute the point estimates (means), and interval estimates
#    # (confidence intervals for means) for 95% confidence level. Do this for
#    # three different configurations (3 preparation rooms, 4 recoveries: 3p 5r,
#    # 4p 5r).
#
#    sample = mk_sampler(variables, slice(-1, 499, -250))
#
#    experiments = [
#        (3, 4),
#        (3, 5),
#        (4, 5),
#    ]
#    N = 20
#
#    experiment_results = []
#    for experiment in experiments:
#        print('experiment', experiment)
#        data = sample_experiment(
#            N, lambda: run_experiment(1000, *experiment,
#                                      patient_generators=[DEFAULT_PG_PARAMS]),
#            variables, sample, reseed=reseed)
#        experiment_results.append(data)
#
#    # save
#    all_data = (N, list(zip(experiments, experiment_results)))
#
#    with open(out_f, "wb") as f:
#        pickle.dump(all_data, f)
#
#
#def tasks1n2():
#    tasks('tasks1n2.dat', ['prep_queue', 'prep_free', 'surg_backed_up'], False)
#
#
#def task3():
#    tasks('task3.dat', ['prep_queue', 'prep_free', 'surg_backed_up'], True)
#
#
#def task4():
#    tasks('task4.dat',
#          ['prep_queue', 'prep_free', 'surg_backed_up', 'recov_full'], True)
#
#
#def debug():
#    tasks('debug.dat',
#          ['prep_free', 'surg_backed_up', 'recov_full', 'recov_free'], True)

def proc_params(flat_options):
    preparation_rooms = flat_options.pop('preparation_rooms')
    recovery_rooms = flat_options.pop('recovery_rooms')
    return {
        'preparation_rooms': preparation_rooms,
        'recovery_rooms': recovery_rooms,
        'patient_generators': (flat_options,),
    }


def get_experiment_params(bits):
    out_dict = {
        'iat': (
            (randunif if bits[0] else randexp,) +
            ((22.5, 20, 25) if bits[1] else (25, 20, 30)))
    }
    PARAM_VARIANTS = [
        ('prep_time', ((randexp, 40, -1, -1), (randunif, 40, 30, 50))),
        ('recovery_time', ((randexp, 40, -1, -1), (randunif, 40, 30, 50))),
        ('preparation_rooms', (4, 5)),
        ('recovery_rooms', (4, 5)),
    ]
    for idx, bit in enumerate(bits[2:]):
        name, values = PARAM_VARIANTS[idx]
        value = values[bit]
        out_dict[name] = value
    return out_dict


EXPERIMENTS = [
    # 1
    (0, 1, 0, 1, 0, 1),
    # 2
    (0, 1, 0, 0, 1, 0),
    # 3
    (0, 0, 1, 0, 0, 1),
    # 4
    (0, 0, 1, 1, 1, 0),
    # 5
    (1, 0, 0, 1, 0, 0),
    # 6
    (1, 0, 0, 0, 1, 1),
    # 7
    (1, 1, 1, 0, 0, 0),
    # 8
    (1, 1, 1, 1, 1, 1)
]


def task2():
    SIMULATION_RUNS = 10
    TOTAL_TIME = 50000

    experiment = DEFAULT_PG_PARAMS.copy()
    experiment.update(get_experiment_params(EXPERIMENTS[6]))

    experiment = proc_params(experiment)

    # Sample 10 times
    i = 0
    data = []
    for i in range(SIMULATION_RUNS):
        print()
        print('Run', i + 1)
        pipeline, monitor = run_experiment(TOTAL_TIME, **experiment)
        data.append(monitor.data['prep_queue'])

    data = numpy.column_stack(data)

    with open('task2.dat', "wb") as f:
        pickle.dump(data, f)


def task3():
    TOTAL_TIME = 20001

    data = []
    for i, experiment_design in enumerate(EXPERIMENTS):
        random.seed(RANDOM_SEED)
        print()
        print('Run', i + 1)
        print('Experiment', experiment_design)
        experiment = DEFAULT_PG_PARAMS.copy()
        experiment.update(get_experiment_params(experiment_design))

        experiment = proc_params(experiment)
        pipeline, monitor = run_experiment(TOTAL_TIME, **experiment)
        data.append((
            i,
            experiment_design,
            monitor.data[['prep_queue', 'surg_backed_up', 'surg_utilisation']]))

    with open('task3.dat', "wb") as f:
        pickle.dump(data, f)


if __name__ == '__main__':
    locals()[sys.argv[1]]()
