package myPolicy;

import java.util.*;

import org.ethz.las.bandit.logs.yahoo.Article;
import org.ethz.las.bandit.logs.yahoo.User;
import org.ethz.las.bandit.policies.ContextualBanditPolicy;
import org.jblas.DoubleMatrix;
import org.jblas.Solve;

public class MyPolicy implements ContextualBanditPolicy<User, Article, Boolean> {
	public static final double ALPHA = 1.0;
	public static final int ARTICLE_COUNT = 271;
	public static final int ARTICLE_FEAT_DIMEN = 6;
	public static final int USER_FEAT_DIMEN = 6;
	private DoubleMatrix ones;
	private DoubleMatrix identity;
	private DoubleMatrix[] matrixA;
	private DoubleMatrix[] vectorB;
	private DoubleMatrix userFeature;
	
  // Here you can load the article features.
  public MyPolicy(String articleFilePath) {
  	matrixA = new DoubleMatrix[ARTICLE_COUNT];
  	vectorB = new DoubleMatrix[ARTICLE_COUNT];
  	ones = DoubleMatrix.ones(ARTICLE_FEAT_DIMEN);
  	identity = DoubleMatrix.diag(ones, ARTICLE_FEAT_DIMEN, ARTICLE_FEAT_DIMEN);
  	userFeature = new DoubleMatrix(USER_FEAT_DIMEN);
  }

  @Override
  public Article getActionToPerform(User visitor, List<Article> possibleActions) {
  	for (int i = 0; i < USER_FEAT_DIMEN; i++) {
  		userFeature.put(i, 0, visitor.getFeatures()[i]);
  	}
  	double max = Double.NEGATIVE_INFINITY;
  	int maxIndex = 0;
  	for (Article article : possibleActions) {
  		int id = article.getID() % ARTICLE_COUNT;
  		if (matrixA[id] == null) {
  			matrixA[id] = DoubleMatrix.diag(ones, ARTICLE_FEAT_DIMEN, ARTICLE_FEAT_DIMEN);
  			vectorB[id] = DoubleMatrix.zeros(ARTICLE_FEAT_DIMEN);
  		}
  		DoubleMatrix inverse = Solve.solve(matrixA[id], identity);
  		DoubleMatrix theta = inverse.mmul(vectorB[id]);
  		DoubleMatrix intermediate = userFeature.transpose().mmul(inverse);
  		double confidenceWidth = ALPHA * Math.sqrt(intermediate.dot(userFeature));
  		double p = theta.transpose().dot(userFeature) + confidenceWidth;
  		if (p > max) {
  			max = p;
  			maxIndex = possibleActions.indexOf(article);
  		}
  		matrixA[id] = matrixA[id].add(userFeature.transpose().mmul(userFeature));
  	}
    return possibleActions.get(maxIndex);
  }

  @Override
  public void updatePolicy(User c, Article a, Boolean reward) {
  	for (int i = 0; i < ARTICLE_COUNT; i++) {
  		if (vectorB[i] != null) {
  			if (reward) {
  				vectorB[i] = vectorB[i].add(userFeature);
  			}
  		}
  	}
  }
}
