package zx.soft.ensemble.selection.simple;

import java.io.FileWriter;

/**
 * A class encapsulating an N-class ensemble.
 * This class is used for N-class classification.
 * 
 * @author wanggang
 *
 */
public class Ensemble extends Model {

	private int n;
	private Predictions[] nClassBag;
	private Predictions nClassPred;

	public Ensemble(Targets[] targets, int size) {
		this.n = targets.length;
		this.numModels = 0;
		for (int i = 0; i < n; i++) {
			this.nClassBag[i] = new Predictions(targets[i]);
		}
		this.nClassPred = new Predictions(targets[0]);
		this.nClassPred.setNumModels(1); // ugly
	}

	public Ensemble(double perf) {
		this.performance = perf;
	}

	@Override
	public Model justPerf() {
		computePerformance();
		return new Ensemble(getPerformance());
	}

	@Override
	public void add(Predictions model) {
		this.nClassBag[model.getPredictedClass()].add(model);
		updateNClassPred();
		this.name = model.getName();
		this.numModels += model.getNumModels();
	}

	@Override
	public void sub(Predictions model) {
		this.nClassBag[model.getPredictedClass()].sub(model);
		updateNClassPred();
		this.name = model.getName();
		this.numModels -= model.getNumModels();
	}

	@Override
	public double compute(int i) {
		return nClassPred.compute(i);
	}

	private void updateNClassPred() {
		int argMax = 0;
		double max = nClassBag[argMax].getProba(0);
		double current;
		for (int i = 0; i < nClassBag[0].getSize(); i++) {
			argMax = 0;
			max = nClassBag[argMax].getProba(i);
			for (int j = 0; j < n; j++) {
				current = nClassBag[j].getProba(i);
				if (current > max) {
					max = current;
					argMax = j;
				}
			}
			nClassPred.setProbaClass(i, max);
			nClassPred.setTarget(i, nClassBag[argMax].getTargets().getTrueValue(i));
		}
	}

	// may have to change that
	@Override
	public void write(FileWriter out) {
		nClassPred.write(out);
	}

	@Override
	public Model copy() {
		return nClassPred.copy();
	}

}