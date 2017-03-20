class IS():
	def __init__(self, stepsNo, burnIn, state0, nominal, importance):
		self.stepsNo = stepsNo
		self.burnIn  = burnIn
		self.state 	 = state0
		self.nominal = nominal
		self.importance = importance
		self.weight_history_importance = []
		self.weight_history_nominal = []
		self.weight_history = []
		self.state_history = []
		
	def run(self):
		for _ in range(self.burnIn):
			self.update()
		for _ in range(self.stepsNo):
			self.update()
			self.getState()

	def update(self):
		self.state = self.importance.rvState()

	def getState(self):
		logP = self.nominal.unnorm_log_pdf( *self.state )
		logQ = self.importance.unnorm_log_pdf( *self.state )
		self.weight_history_importance.append( logQ )
		self.weight_history_nominal.append( logP )
		self.weight_history.append( logP - logQ )
		self.state_history.append( self.state )