# coding=utf-8
# Copyright 2021 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Customized (Adam H.) BART model configuration"""

from transformers import BartConfig

class BartSpektroConfig(BartConfig):
    """
        max_log_id (`int`, defaults to `None` - Adam's customizaiton):
            If not None, positional embeddings up to this value can be trained (if provided) and summed with standard embeddings. 
    """
    def __init__(
        self,
        max_log_id=None,
        **kwargs):
        
        self.max_log_id=max_log_id
        super().__init__(
            **kwargs,
        )