package myPolicy;

import java.io.IOException;
import java.util.Hashtable;
import java.util.List;
import java.util.Random;

import org.ethz.las.bandit.logs.yahoo.Article;
import org.ethz.las.bandit.logs.yahoo.ArticleFeatures;
import org.ethz.las.bandit.logs.yahoo.User;
import org.ethz.las.bandit.policies.ContextualBanditPolicy;
import org.jblas.DoubleMatrix;
import org.jblas.Solve;

public class MyPolicy implements ContextualBanditPolicy<User, Article, Boolean> {
	
	// confidence is 1 - DELTA, DELTA smaller, larger interval
	public static final double DELTA = 0.01;
	public static final double ALPHA = 1 + Math.sqrt(Math.log(2 / DELTA) / 2);
	
	// update policy with this interval
	public static final int UPDATE_INTERVAL = 1000;
	
	// specified in the project description
	public static final int ARTICLE_COUNT = 271;
	public static final int ARTICLE_FEAT_DIMEN = 6;
	public static final int USER_FEAT_DIMEN = 6;
	public static final int K_CONST = USER_FEAT_DIMEN;//k
	
	// threshold to compare double values
	private static final double THRESHOLD = 0.00000000001;
	// max article age to contribute to feedback (milliseconds)
	private static final long AGE_THRESHOLD = Long.MAX_VALUE;
	private DoubleMatrix ones;
	private DoubleMatrix identity;
	private DoubleMatrix[] matrixAt;
	private DoubleMatrix[] vectorbt;
	private DoubleMatrix userFeature;
	private DoubleMatrix articleFeature;
	private long[] birthTimes;
	private Random random;
	private int logLinesToUpdate = UPDATE_INTERVAL;
	//private int chosenID;
	
	
	//For Hybrid Model
	private DoubleMatrix matrixA0;
	private DoubleMatrix vectorb0;
	private DoubleMatrix[] matrixB;
	
	private DoubleMatrix vectorBeta;
	//private DoubleMatrix vectorTheta;
	
	private Hashtable<Integer, ArticleFeatures> articleFeatureTable;
	
  // Here you can load the article features.
  public MyPolicy(String articleFilePath) throws IOException {
  	matrixAt = new DoubleMatrix[ARTICLE_COUNT];
  	vectorbt = new DoubleMatrix[ARTICLE_COUNT];
  	matrixB = new DoubleMatrix[ARTICLE_COUNT];
  	
  	ones = DoubleMatrix.ones(ARTICLE_FEAT_DIMEN);
  	identity = DoubleMatrix.diag(ones, ARTICLE_FEAT_DIMEN, ARTICLE_FEAT_DIMEN);
  	userFeature = new DoubleMatrix(USER_FEAT_DIMEN);
  	articleFeature = new DoubleMatrix(ARTICLE_FEAT_DIMEN);
  	random = new Random();
  	
  	//For Hybrid Model
  	matrixA0 = DoubleMatrix.diag(ones, K_CONST, K_CONST);
  	vectorb0 = DoubleMatrix.zeros(K_CONST);
  	
  	ArticleReader aReader = new ArticleReader(articleFilePath);
  	//read in article features
  	articleFeatureTable = aReader.read();
  	
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

		//calculate beta
  	DoubleMatrix matrixA0inversed = inversed(matrixA0);
		vectorBeta = matrixA0inversed.dup().mmul(vectorb0);
  	
		int chosenID = 0;
  	// loop through availabe articles
  	for (Article article : possibleActions) {
  		// mod because article ID is not in the range of [0..270]
  		int id = article.getID() % ARTICLE_COUNT;

  		if (matrixAt[id] == null) {
  			// initialize if article is new
  			matrixAt[id] = DoubleMatrix.diag(ones, ARTICLE_FEAT_DIMEN, ARTICLE_FEAT_DIMEN);
  			vectorbt[id] = DoubleMatrix.zeros(ARTICLE_FEAT_DIMEN);
  			birthTimes[id] = visitor.getTimestamp();
  			
  			//For Hybrid Model
  			matrixB[id] = DoubleMatrix.zeros(ARTICLE_FEAT_DIMEN, K_CONST);
  		}
  		
  		//get articleFeature, using real ID;
  		ArticleFeatures aF = articleFeatureTable.get(article.getID());
  		for (int i = 0; i < ARTICLE_FEAT_DIMEN; i++) {
				double[] aFArr = aF.getFeatures();
				articleFeature.put(i, aFArr[i]);
			}
  		
  		// calculate the inverse by solving AX=I
  		DoubleMatrix matrixAtinversed = inversed(matrixAt[id]);
  		
  		
  		// calculate theta
  		DoubleMatrix tempforTheta=matrixB[id].dup().mmul(vectorBeta);
  		DoubleMatrix vectorTheta = matrixAtinversed.dup().mmul(vectorbt[id].dup().sub(
  													tempforTheta));
  		
  		//s_{t,a}
  		DoubleMatrix userFeatureTranspose = userFeature.transpose();
  		DoubleMatrix articleFeatureTranspose = articleFeature.transpose();
  		
  		DoubleMatrix common24 = matrixA0inversed.dup().mmul(matrixB[id].transpose())
					.mmul(matrixAtinversed).mmul(userFeature);
			DoubleMatrix common34 = userFeatureTranspose.dup().mmul(matrixAtinversed);
			double s1 = articleFeatureTranspose.dot(matrixA0inversed.mmul(userFeature));
			double s2 = articleFeatureTranspose.dot(common24);
			double s3 = common34.dot(userFeature);
			double s4 = common34.dot(matrixB[id].mmul(common24));
  		
  		double s = s1-2*s2+s3+s4;
  		
  		//calculate P
  		double p = articleFeatureTranspose.dot(vectorBeta)
  					+ userFeatureTranspose.dot(vectorTheta)
  					+ ALPHA*Math.sqrt(s);
  		
  		// pick this article if p value is greater than max
  		if (p - max > THRESHOLD) {
  			max = p;
  			maxIndex = possibleActions.indexOf(article);
  			chosenID = article.getID();
  		}
  	}
  	
  	logLinesToUpdate--;
  	if (logLinesToUpdate == 0) {
	  	int id = chosenID % ARTICLE_COUNT;
	  	DoubleMatrix chosenArticleFeature = new DoubleMatrix(articleFeatureTable.get(chosenID).getFeatures());
	  	DoubleMatrix matrixBTranspose = matrixB[id].transpose();
	  	DoubleMatrix matrixAtinversed = inversed(matrixAt[id]);
	
	  	//////update shared part1////
	  	matrixA0 = matrixA0.add(matrixBTranspose.dup().mmul(matrixAtinversed).mmul(matrixB[id]));
	  	vectorb0 = vectorb0.add(matrixBTranspose.dup().mmul(matrixAtinversed).mmul(vectorbt[id]));
	  	
	  	//////update separate part////
	  	matrixAt[id] = matrixAt[id].add(userFeature.dup().mmul(userFeature.transpose()));
	  	matrixB[id] = matrixB[id].add(userFeature.dup().mmul(chosenArticleFeature.transpose()));

	  	
	  	//////update shared part2////
	  	matrixAtinversed = inversed(matrixAt[id]);
	  	matrixBTranspose = matrixB[id].transpose();
	  	matrixA0 = matrixA0.add(chosenArticleFeature.dup().mmul(chosenArticleFeature.transpose()))
	  						.sub(matrixBTranspose.dup().mmul(matrixAtinversed).mmul(matrixB[id]));
	  	vectorb0 = vectorb0.sub(matrixBTranspose.dup().mmul(matrixAtinversed).mmul(vectorbt[id]));

	  	
	  	// reset interval
	  	logLinesToUpdate = UPDATE_INTERVAL;
  	}
  	
    return possibleActions.get(maxIndex);
  }

  private DoubleMatrix inversed(DoubleMatrix matrix)
  {
	  return Solve.solve(matrix, identity);
  }
  
  @Override
  public void updatePolicy(User c, Article a, Boolean reward) {
  	// convert userFeature into DoubleMatrix
  	for (int i = 0; i < USER_FEAT_DIMEN; i++) {
  		userFeature.put(i, 0, c.getFeatures()[i]);
  	}
  	
  	int id = a.getID() % ARTICLE_COUNT;
  		
  	// check article age
  	if (c.getTimestamp() - birthTimes[id] <= AGE_THRESHOLD) {
			DoubleMatrix chosenArticleFeature = new DoubleMatrix(articleFeatureTable.get(a.getID()).getFeatures());
    	if (reward) {
    		vectorbt[id] = vectorbt[id].add(userFeature);
    	}
    	
    	if(reward)
    	{
    		vectorb0 = vectorb0.add(chosenArticleFeature);
    	}
  	}
  }

}

