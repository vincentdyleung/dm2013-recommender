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
	private static final long AGE_THRESHOLD = Long.MAX_VALUE;
	private DoubleMatrix ones;
	private DoubleMatrix identity;
	private Hashtable<Integer, DoubleMatrix> matrixA;
	private Hashtable<Integer, DoubleMatrix> vectorB;
	private DoubleMatrix userFeature;
	private Hashtable<Integer, Long> birthTimes;
	private Random random;
	private Integer chosenID;
	
  // Here you can load the article features.
  public MyPolicy(String articleFilePath) {
  	matrixA = new Hashtable<Integer, DoubleMatrix>(ARTICLE_COUNT);
  	vectorB = new Hashtable<Integer, DoubleMatrix>(ARTICLE_COUNT);
  	ones = DoubleMatrix.ones(ARTICLE_FEAT_DIMEN);
  	identity = DoubleMatrix.diag(ones, ARTICLE_FEAT_DIMEN, ARTICLE_FEAT_DIMEN);
  	userFeature = new DoubleMatrix(USER_FEAT_DIMEN);
  	random = new Random();
  	birthTimes = new Hashtable<Integer, Long>(ARTICLE_COUNT);
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
  		Integer articleID = article.getID();
  		// check if article is new
  		if (matrixA.get(articleID) == null) {
  			// initialize if article is new
  			matrixA.put(articleID, identity.dup());
  			DoubleMatrix zeros = DoubleMatrix.zeros(ARTICLE_FEAT_DIMEN);
  			vectorB.put(articleID, zeros);
  			birthTimes.put(articleID, visitor.getTimestamp());
  		}
  		
  		// calculate the inverse by solving AX=I
  		DoubleMatrix inverse = Solve.solve(matrixA.get(articleID), identity);
  		
  		// calculate theta
  		DoubleMatrix theta = inverse.dup().mmul(vectorB.get(articleID));
  		
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
  	//update the matrix
  	matrixA.get(chosenID).add(userFeature.dup().mmul(userFeature.transpose()));
    return possibleActions.get(maxIndex);
  }

  @Override
  public void updatePolicy(User c, Article a, Boolean reward) {
  	// convert userFeature into DoubleMatrix
  	for (int i = 0; i < USER_FEAT_DIMEN; i++) {
  		userFeature.put(i, 0, c.getFeatures()[i]);
  	}
  	
  	Integer id = a.getID();
  	
  	// check article age
  	if (c.getTimestamp() - birthTimes.get(id) <= AGE_THRESHOLD) {
  		
    	
    	
    	//update the vector if clicked through
    	if (reward) {
    		vectorB.get(id).add(userFeature);
    	}
  	}
  	
  }
}
