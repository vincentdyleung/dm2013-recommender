package myPolicy;

import java.util.*;

import org.ethz.las.bandit.logs.yahoo.Article;
import org.ethz.las.bandit.logs.yahoo.User;
import org.ethz.las.bandit.policies.ContextualBanditPolicy;
import org.jblas.DoubleMatrix;
import org.jblas.Solve;

public class MyPolicy implements ContextualBanditPolicy<User, Article, Boolean> {
	
	// confidence is 1 - DELTA, DELTA smaller, larger interval
	public static final double DELTA = 0.01;
	public static final double ALPHA = 1 + Math.sqrt(Math.log(2 / DELTA) / 2);
	
	// specified in the project description
	public static final int ARTICLE_COUNT = 271;
	public static final int ARTICLE_FEAT_DIMEN = 6;
	public static final int USER_FEAT_DIMEN = 6;
	
	// threshold to compare double values
	private static final double THRESHOLD = 0.00000000001;
	// max article age to contribute to feedback (milliseconds)
	private static final long AGE_THRESHOLD = 10800000;
	private DoubleMatrix ones;
	private DoubleMatrix identity;
	private DoubleMatrix[] matrixA;
	private DoubleMatrix[] vectorB;
	private DoubleMatrix userFeature;
	private long[] birthTimes;
	private Random random;
	private int chosenID;
	
  // Here you can load the article features.
  public MyPolicy(String articleFilePath) {
  	matrixA = new DoubleMatrix[ARTICLE_COUNT];
  	vectorB = new DoubleMatrix[ARTICLE_COUNT];
  	ones = DoubleMatrix.ones(ARTICLE_FEAT_DIMEN);
  	identity = DoubleMatrix.diag(ones, ARTICLE_FEAT_DIMEN, ARTICLE_FEAT_DIMEN);
  	userFeature = new DoubleMatrix(USER_FEAT_DIMEN);
  	random = new Random();
  	birthTimes = new long[ARTICLE_COUNT];
  }

  @Override
  public Article getActionToPerform(User visitor, List<Article> possibleActions) {
  	// convert userFeature into DoubleMatrix
  	for (int i = 0; i < USER_FEAT_DIMEN; i++) {
  		userFeature.put(i, 0, visitor.getFeatures()[i]);
  	}
  	double max = Double.NEGATIVE_INFINITY;
  	int maxIndex = random.nextInt(possibleActions.size());
  	
  	// loop through availabe articles
  	for (Article article : possibleActions) {
  		// mod because article ID is not in the range of [0..270]
  		int id = article.getID() % ARTICLE_COUNT;
  		
  		// check if article is new
  		if (matrixA[id] == null) {
  			// initialize if article is new
  			matrixA[id] = DoubleMatrix.diag(ones, ARTICLE_FEAT_DIMEN, ARTICLE_FEAT_DIMEN);
  			vectorB[id] = DoubleMatrix.zeros(ARTICLE_FEAT_DIMEN);
  			birthTimes[id] = visitor.getTimestamp();
  		}
  		
  		// calculate the inverse by solving AX=I
  		DoubleMatrix inverse = Solve.solve(matrixA[id], identity);
  		
  		// calculate theta
  		DoubleMatrix theta = inverse.mmul(vectorB[id]);
  		
  		// break calculation of p into three steps
  		DoubleMatrix intermediate = userFeature.transpose().mmul(inverse);
  		double confidenceWidth = ALPHA * Math.sqrt(intermediate.dot(userFeature));
  		double p = theta.transpose().dot(userFeature) + confidenceWidth;
  		
  		// pick this article if p value is greater than max
  		if (p - max > THRESHOLD) {
  			max = p;
  			maxIndex = possibleActions.indexOf(article);
  			chosenID = article.getID();
  		}
  	}
    return possibleActions.get(maxIndex);
  }

  @Override
  public void updatePolicy(User c, Article a, Boolean reward) {
  	// convert userFeature into DoubleMatrix
  	for (int i = 0; i < USER_FEAT_DIMEN; i++) {
  		userFeature.put(i, 0, c.getFeatures()[i]);
  	}
  	
  	int id = chosenID % ARTICLE_COUNT;
  	
  	// check article age
  	if (c.getTimestamp() - birthTimes[id] <= AGE_THRESHOLD) {
  		//update the matrix
    	matrixA[id] = matrixA[id].add(userFeature.mmul(userFeature.transpose()));
    	
    	//update the vector if clicked through
    	if (reward) {
    		vectorB[id] = vectorB[id].add(userFeature);
    	}
  	}
  	
  }
}
