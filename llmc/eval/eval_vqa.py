import random
from typing import List, Optional, Union

import numpy as np
import torch
from lmms_eval.evaluator import evaluate
from lmms_eval.evaluator_utils import run_task_tests
from lmms_eval.loggers.evaluation_tracker import EvaluationTracker
from lmms_eval.tasks import TaskManager, get_task_dict
from lmms_eval.utils import (get_datetime_str, make_table,
                             simple_parse_args_string)
from loguru import logger

from llmc.utils.registry_factory import MODEL_REGISTRY


class VQAEval:
    def __init__(self, config):
        self.eval_config = config.eval
        self.model_path = config.model.path
        self.dataset = self.eval_config['name']
        if not isinstance(self.dataset, list):
            self.dataset = [self.dataset, ]
        self.eval_dataset_path = self.eval_config['path']
        self.eval_bs = self.eval_config['bs']

    def eval(
        self,
        llmc_model,
        model_args: Optional[Union[str, dict]] = None,
        tasks: Optional[List[Union[str, dict, object]]] = None,
        num_fewshot: Optional[int] = None,
        batch_size: Optional[Union[int, str]] = None,
        max_batch_size: Optional[int] = None,
        device: Optional[str] = None,
        use_cache: Optional[str] = None,
        cache_requests: bool = False,
        rewrite_requests_cache: bool = False,
        delete_requests_cache: bool = False,
        limit: Optional[Union[int, float]] = None,
        bootstrap_iters: int = 100000,
        check_integrity: bool = False,
        write_out: bool = False,
        log_samples: bool = True,
        evaluation_tracker: Optional[EvaluationTracker] = None,
        system_instruction: Optional[str] = None,
        apply_chat_template: bool = False,
        fewshot_as_multiturn: bool = False,
        gen_kwargs: Optional[str] = None,
        task_manager: Optional[TaskManager] = None,
        verbosity: str = 'INFO',
        predict_only: bool = False,
        random_seed: int = 0,
        numpy_random_seed: int = 1234,
        torch_random_seed: int = 1234,
        fewshot_random_seed: int = 1234,
        datetime_str: str = get_datetime_str(),
        cli_args=None,
    ):

        model = llmc_model.eval_name
        model_args = 'pretrained=' + self.model_path + ',device_map=auto'
        batch_size = self.eval_bs
        tasks = self.dataset
        num_fewshot = 0

        seed_message = []
        if random_seed is not None:
            # See https://github.com/EleutherAI/lm-evaluation-harness/pull/1412
            seed_message.append(f'Setting random seed to {random_seed}')
            random.seed(random_seed)

        if numpy_random_seed is not None:
            seed_message.append(f'Setting numpy seed to {numpy_random_seed}')
            np.random.seed(numpy_random_seed)

        if torch_random_seed is not None:
            seed_message.append(f'Setting torch manual seed to {torch_random_seed}')
            torch.manual_seed(torch_random_seed)

        if seed_message:
            logger.info(' | '.join(seed_message))

        assert tasks != [], 'No tasks specified, or no tasks found. Please verify the task names.'

        if gen_kwargs:
            gen_kwargs = simple_parse_args_string(gen_kwargs)
            logger.warning('generation_kwargs specified through cli.')
            if gen_kwargs == '':
                gen_kwargs = None

        if model_args is None:
            model_args = ''

        if task_manager is None:
            task_manager = TaskManager(verbosity, model_name=model)

        task_dict = get_task_dict(tasks, task_manager)

        lm = MODEL_REGISTRY[model].create_from_arg_string(
            model_args,
            {
                'llmc_model': llmc_model.vlm_model,
                'batch_size': batch_size,
                'device': device,
            },
        )
        # helper function to recursively apply config overrides to leaf subtasks,
        # skipping their constituent groups.
        # (setting of num_fewshot ; bypassing metric calculation ; setting fewshot seed)

        def _adjust_config(task_dict):
            adjusted_task_dict = {}
            for task_name, task_obj in task_dict.items():
                if isinstance(task_obj, dict):
                    adjusted_task_dict = {
                        **adjusted_task_dict,
                        **{task_name: _adjust_config(task_obj)},
                    }

                else:
                    task_obj = task_dict[task_name]
                    if type(task_obj) == tuple:
                        group, task_obj = task_obj
                        if task_obj is None:
                            continue
                    lm.task_dict[task_name] = task_obj.dataset
                    if 'generate_until' in task_obj.get_config('output_type'):
                        if gen_kwargs is not None:
                            task_obj.set_config(key='generation_kwargs',
                                                value=gen_kwargs, update=True)

                    if predict_only:
                        logger.info(f'Processing {task_name} in output-only mode. \
                                    Metrics will not be calculated!')
                        # we have to change the class properties post-hoc. This is pretty hacky.
                        task_obj.override_metric(metric_name='bypass')

                    # override tasks' fewshot values to
                    # the provided num_fewshot arg value
                    # except if tasks have it set to 0 manually in their configs--then
                    # we should never overwrite that
                    if num_fewshot is not None:
                        if (default_num_fewshot := task_obj.get_config('num_fewshot')) == 0:
                            logger.info(f'num_fewshot has been set to 0 for {task_name} \
                                        in its config. Manual configuration will be ignored.')
                        else:
                            logger.warning(f'Overwriting default num_fewshot of {task_name} \
                                           from {default_num_fewshot} to {num_fewshot}')
                            task_obj.set_config(key='num_fewshot', value=num_fewshot)
                    else:
                        # if num_fewshot not provided, and the task does not define a default one,
                        # default to 0
                        if (default_num_fewshot := task_obj.get_config('num_fewshot')) is None:
                            task_obj.set_config(key='num_fewshot', value=0)
                    # fewshot_random_seed set for tasks, even with a default num_fewshot
                    # (e.g. in the YAML file)
                    task_obj.set_fewshot_seed(seed=fewshot_random_seed)
                    # logger.info(f"Setting fewshot random generator seed to {fewshot_random_seed}")

                    adjusted_task_dict[task_name] = task_obj

            return adjusted_task_dict

        task_dict = _adjust_config(task_dict)

        if check_integrity:
            run_task_tests(task_list=tasks)

        if evaluation_tracker is not None:
            evaluation_tracker.general_config_tracker.log_experiment_args(
                model_source=model,
                model_args=model_args,
                system_instruction=system_instruction,
                chat_template=lm.chat_template if apply_chat_template else None,
                fewshot_as_multiturn=fewshot_as_multiturn,
            )

        results = evaluate(
            lm=lm,
            task_dict=task_dict,
            limit=limit,
            cache_requests=cache_requests,
            rewrite_requests_cache=rewrite_requests_cache,
            bootstrap_iters=bootstrap_iters,
            write_out=write_out,
            log_samples=True if predict_only else log_samples,
            system_instruction=system_instruction,
            apply_chat_template=apply_chat_template,
            fewshot_as_multiturn=fewshot_as_multiturn,
            verbosity=verbosity,
            cli_args=cli_args,
        )

        if hasattr(lm, '_model'):
            del lm._model
            torch.cuda.empty_cache()

        if lm.rank == 0:
            if isinstance(model, str):
                model_name = model
            elif hasattr(model, 'config') and hasattr(model.config, '_name_or_path'):
                model_name = model.config._name_or_path
            else:
                model_name = type(model).__name__

            # add info about the model and few shot config
            results['config'] = {
                'model': model_name,
                'model_args': model_args,
            }
            # add more detailed model info if available TODO: add model info
            # if isinstance(lm, lm_eval.models.huggingface.HFLM):
            #     results["config"].update(lm.get_model_info())
            # add info about execution
            results['config'].update(
                {
                    'batch_size': batch_size,
                    'batch_sizes': (list(lm.batch_sizes.values())
                                    if hasattr(lm, 'batch_sizes') else []),
                    'device': device,
                    'use_cache': use_cache,
                    'limit': limit,
                    'bootstrap_iters': bootstrap_iters,
                    'gen_kwargs': gen_kwargs,
                    'random_seed': random_seed,
                    'numpy_seed': numpy_random_seed,
                    'torch_seed': torch_random_seed,
                    'fewshot_seed': fewshot_random_seed,
                }
            )
            results['date'] = datetime_str
            # add_env_info(results)  # additional environment info to results
            # add_tokenizer_info(results, lm)  # additional info about tokenizer
            return '\n' + make_table(results)
        else:
            return None
