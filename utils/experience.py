import sys
import time 
import numpy as np
from collections import deque


class Experience:
    def __init__(self, 
                max_size=None, 
                prompt_ids=None, 
                completion_ids=None,
                old_logps=None, 
                ref_logps=None, 
                advantage = None,
                weight=None, 
                count=None,
                other_args=None):
        self.max_size = max_size
        
        self.prompt_ids = deque(prompt_ids if prompt_ids is not None else [], maxlen=max_size)
        self.completion_ids = deque(completion_ids if completion_ids is not None else [], maxlen=max_size)
        self.old_logps = deque(old_logps if old_logps is not None else [], maxlen=max_size)
        self.ref_logps = deque(ref_logps if ref_logps is not None else [], maxlen=max_size)
        self.advantage = deque(advantage if advantage is not None else [], maxlen=max_size)
        self.weight = deque(weight if weight is not None else [], maxlen=max_size)
        self.count = deque(count if count is not None else [], maxlen=max_size)
        self.other_args = deque(other_args if other_args is not None else [], maxlen=max_size)
        
    def length(self):
        return len(self.prompt_ids)
    
    def get_memory_usage(self):
        total_memory = 0
        attrs = ['prompt_ids', 'completion_ids', 'old_logps', 'ref_logps', 
                'advantage', 'weight', 'count', 'other_args']
        for attr_name in attrs:
            attr = getattr(self, attr_name)
            total_memory += sys.getsizeof(attr)
            for item in attr:
                total_memory += sys.getsizeof(item)

        return total_memory

    def push_sample_batch(self, *args, **kwargs):
        for batch in zip(*args):
            self.push_sample(batch)

    def push_sample(self, batch):
        prompt_ids, completion_ids, old_logps, ref_logps, advantage, weight, count, other_args = batch

        default_values = {
            "prompt_ids": ["null"],
            "completion_ids": ["null"],
            "old_logps": ["null"],
            "ref_logps": ["null"],
            "advantage": ["null"],
            "weight": ["null"],
            "count": ["null"],
            "other_args": ["null"]
        }
        
        fields = [
            ("prompt_ids", prompt_ids),
            ("completion_ids", completion_ids),
            ("old_logps", old_logps),
            ("ref_logps", ref_logps),
            ("advantage", advantage),
            ("weight", weight),
            ("count", count),
            ("other_args", other_args)
        ]
        
        lengths = []
        for name, value in fields:
            container = getattr(self, name)
            if value is not None:
                container.append(value)
            else:
                container.append(default_values[name])
            lengths.append(len(container))
        
        if len(set(lengths)) > 1:
            raise ValueError(
                f"Inconsistent argument lengths: "
                f"prompt_ids={lengths[0]}, "
                f"completion_ids={lengths[1]}, "
                f"old_logps={lengths[2]}, "
                f"ref_logps={lengths[3]}, "
                f"advantage={lengths[4]}, "
                f"weight={lengths[5]}, "
                f"count={lengths[6]}, "
                f"other_args={lengths[7]}"
            )
    def delay(self, a_decay=0.9, samples_per_step=None):
        available_size = len(self.advantage)
        if available_size == 0:
            return
        samples_per_step = self._normalize_samples_per_step(samples_per_step, available_size)
        for i in range(len(self.advantage)):
            self.advantage[i] = self._decayed_advantage(i, available_size, samples_per_step, a_decay)

    def _normalize_samples_per_step(self, samples_per_step, batch_size):
        if samples_per_step is None:
            samples_per_step = batch_size
        samples_per_step = int(samples_per_step)
        if samples_per_step <= 0:
            raise ValueError("samples_per_step must be positive")
        return samples_per_step

    def _delay_steps(self, index, total_size, samples_per_step):
        return max((total_size - 1 - int(index)) // samples_per_step, 0)

    def _decayed_advantage(self, index, total_size, samples_per_step, a_decay):
        index = int(index)
        delay_steps = self._delay_steps(index, total_size, samples_per_step)
        return self.advantage[index] * (a_decay ** delay_steps)

    def pull_sample(self, batch_size=1, reuse=False, random=False, per=False, per_gamma=0.5, a_decay=0.9, samples_per_step=None, temperature=0.5, logger=None):
        if batch_size <= 0 or not self.prompt_ids:
            return None
        
        available_size = len(self.prompt_ids)
        samples_per_step = self._normalize_samples_per_step(samples_per_step, batch_size)

        if per:
            start_time = time.time()
            if not hasattr(self, 'weight') or len(self.weight) == 0:
                raise ValueError("No weight data is available")
            
            weights = np.array(self.weight)
            sorted_indices = np.argsort(weights[:available_size])
            indices = sorted_indices[-batch_size:]
            np.random.shuffle(indices)

            exp_pulled = {
                'prompt_ids': [self.prompt_ids[i] for i in indices],
                'completion_ids': [self.completion_ids[i] for i in indices],
                'old_logps': [self.old_logps[i] for i in indices],
                'ref_logps': [self.ref_logps[i] for i in indices],
                'advantage': [self._decayed_advantage(i, available_size, samples_per_step, a_decay) for i in indices],
                'weight': [self.weight[i] for i in indices],
                'count': [self.count[i] for i in indices],
                'other_args': [self.other_args[i] for i in indices]
            }

            if reuse:
                tmp_weight = []
                for i in indices:
                    tmp_weight.append(self.weight[i])
                    if self.weight[i] > 100:
                        self.weight[i] -= 120
                    self.weight[i] *= per_gamma
                    self.count[i] += 1
                if logger: logger.info(f"exp pull weight: {tmp_weight} \n max {max(tmp_weight)} min {min(tmp_weight)} mean {np.mean(tmp_weight)} \n index {indices} \n sampel count {np.mean(self.count)}")
                if logger:
                    delay_steps = [self._delay_steps(i, available_size, samples_per_step) for i in indices]
                    logger.info(f"exp delay advantage, samples_per_step {samples_per_step}, delay_steps {delay_steps}, max {max(exp_pulled['advantage'])}, min {min(exp_pulled['advantage'])}")
            else:
                indices.sort(reverse=True)
                for i in indices:
                    del self.prompt_ids[i]
                    del self.completion_ids[i]
                    del self.old_logps[i]
                    del self.ref_logps[i]
                    del self.advantage[i]
                    del self.weight[i]
                    del self.count[i]
                    del self.other_args[i]
            if logger: logger.info(f"per cost time: {time.time()-start_time:.2f}")
        else:  
            if random:
                indices = np.random.choice(available_size, batch_size, replace=False).tolist()
                indices.sort(reverse=True)
                tmp_weight = []
                for i in indices:
                    tmp_weight.append(self.weight[i])
                if logger: logger.info(f"random exp pull weight: {tmp_weight} \n max {max(tmp_weight)} min {min(tmp_weight)} mean {np.mean(tmp_weight)} \n index {indices}")    
                
            else:
                indices = list(range(batch_size))
                indices.sort(reverse=True)

            exp_pulled = {
                'prompt_ids': [self.prompt_ids[i] for i in indices],
                'completion_ids': [self.completion_ids[i] for i in indices],
                'old_logps': [self.old_logps[i] for i in indices],
                'ref_logps': [self.ref_logps[i] for i in indices],
                'advantage': [self.advantage[i] for i in indices],
                'weight': [self.weight[i] for i in indices],
                'count': [self.count[i] for i in indices],
                'other_args': [self.other_args[i] for i in indices]
            }
            
            if logger:
                delay_steps = [self._delay_steps(i, available_size, samples_per_step) for i in indices]
                logger.info(f"random, exp delay advantage, samples_per_step {samples_per_step}, delay_steps {delay_steps}, max {max(exp_pulled['advantage'])}, min {min(exp_pulled['advantage'])}")
                
            if not reuse:
                for i in indices:
                    del self.prompt_ids[i]
                    del self.completion_ids[i]
                    del self.old_logps[i]
                    del self.ref_logps[i]
                    del self.advantage[i]
                    del self.weight[i]
                    del self.count[i]
                    del self.other_args[i]
            
        return exp_pulled

    def get_prompt_ids(self, index=None):
        return self.prompt_ids[index]

    def is_empty(self):
        return len(self.prompt_ids) == 0

    def clear(self):
        self.prompt_ids.clear()
        self.completion_ids.clear()
        self.old_logps.clear()
        self.ref_logps.clear()
        self.advantage.clear()
        self.weight.clear()
        self.count.clear()
        self.other_args.clear()
        
