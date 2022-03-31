"""
This module contains the replay memory class

Authors: Dominik Brockmann, Niklas Schemmer
"""
import random

class ReplayMemory():
	"""
	The replay memory saves seen experiences and can provide randomly sampled batches.
	"""
	def __init__(self, max_amount: int):
		"""
		Init the replay memory.

		Parameter max_amount: The maximum amount of experience items to be stored
		"""
		self.max_amount = max_amount
		# The memory array itself
		self.memory = []
		# Increased for every push into the memory
		self.push_count = 0

	def add(self, experience: tuple):
		"""
		Adds an experience to the replay memory.

		This adds another element to the replay buffer.
		It overrides the first element if the max_amount is reached.

		Parameter experience: The experience element to be added
		"""
		# If the max max_amount is not reached yet the element is appended to the memory
		if len(self.memory)<self.max_amount:
			self.memory.append(experience)
		else:
			# Else the oldest element is overwritten
			self.memory[self.push_count % self.max_amount] = experience
		self.push_count += 1

	def random_sample(self, batch_size: int):
		"""
		Randomly creates a batch of samples from the buffer.

		Parameter batch_size: The size of the batch to be sampled
		"""
		return random.sample(self.memory, batch_size)

	def has_batch_length(self, batch_size: int):
		"""
		Returns if the memory if big enough to sample.

		Parameter batch_size: The size for the batch
		"""
		return len(self.memory) >= batch_size