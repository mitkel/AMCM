class MCalgo:
	def __init__(self, stepsNo, burnIn, target, state0):
		self.stepsNo = stepsNo
		self.burnIn  = burnIn
		self.target  = target
		self.state   = state0

	def run(self):
		for _ in range(self.burnIn):
			self.updateState()
			# print(self.state)
		for _ in range(self.stepsNo):
			self.updateState()
			yield self.state

	def update(self):
		raise NotImplementedError
