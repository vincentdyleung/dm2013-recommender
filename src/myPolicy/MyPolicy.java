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
	public static final int UPDATE_INTERVAL = 100000;
	
	// specified in the project description
	public static final int ARTICLE_COUNT = 271;
	public static final int ARTICLE_FEAT_DIMEN = 6;
	public static final int USER_FEAT_DIMEN = 6;
	public static final int K_CONST = USER_FEAT_DIMEN * ARTICLE_FEAT_DIMEN;//k
	
	// threshold to compare double values
	private static final double THRESHOLD = 0.00000000001;
	// max article age to contribute to feedback (milliseconds)
	private static final long AGE_THRESHOLD = Long.MAX_VALUE;
	private DoubleMatrix dOnes;
	private DoubleMatrix kOnes;
	private DoubleMatrix kIdentity;
	private DoubleMatrix dIdentity;
	private Hashtable<Integer, DoubleMatrix> matrixAt;
	private Hashtable<Integer, DoubleMatrix> vectorbt;
	private DoubleMatrix userFeature;
	private DoubleMatrix articleFeature;
	private Hashtable<Integer, Long> birthTimes;
	private Random random;
	private int logLinesToUpdate = UPDATE_INTERVAL;
	//private int chosenID;
	private DoubleMatrix chosenZed;
	
	
	//For Hybrid Model
	private DoubleMatrix matrixA0;
	private DoubleMatrix vectorb0;
	private Hashtable<Integer, DoubleMatrix> matrixB;
	
	private DoubleMatrix vectorBeta;
	//private DoubleMatrix vectorTheta;
	
	private Hashtable<Integer, ArticleFeatures> articleFeatureTable;
	
  // Here you can load the article features.
  public MyPolicy(String articleFilePath) throws IOException {
  	matrixAt = new Hashtable<Integer, DoubleMatrix>(ARTICLE_COUNT);
  	vectorbt = new Hashtable<Integer, DoubleMatrix>(ARTICLE_COUNT);
  	matrixB = new Hashtable<Integer, DoubleMatrix>(ARTICLE_COUNT);
  	
  	dOnes = DoubleMatrix.ones(USER_FEAT_DIMEN);
  	kOnes = DoubleMatrix.ones(K_CONST);
  	dIdentity = DoubleMatrix.diag(dOnes, USER_FEAT_DIMEN, USER_FEAT_DIMEN);
  	kIdentity = DoubleMatrix.diag(kOnes, K_CONST, K_CONST);
  	userFeature = new DoubleMatrix(USER_FEAT_DIMEN);
  	articleFeature = new DoubleMatrix(ARTICLE_FEAT_DIMEN);
  	random = new Random();
  	
  	//For Hybrid Model
  	DoubleMatrix kOnes = DoubleMatrix.ones(K_CONST);
  	matrixA0 = DoubleMatrix.diag(kOnes, K_CONST, K_CONST);
  	vectorb0 = DoubleMatrix.zeros(K_CONST);
  	
  	ArticleReader aReader = new ArticleReader(articleFilePath);
  	//read in article features
  	articleFeatureTable = aReader.read();
  	
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

		//calculate beta
  	DoubleMatrix matrixA0inversed = kInverse(matrixA0);
		vectorBeta = matrixA0inversed.dup().mmul(vectorb0);
  	
		Integer chosenID = possibleActions.get(random.nextInt(possibleActions.size())).getID();
  	// loop through availabe articles
  	for (Article article : possibleActions) {
  		// mod because article ID is not in the range of [0..270]
  		Integer id = article.getID();

  		if (matrixAt.get(id) == null) {
  			// initialize if article is new
  			matrixAt.put(id, dIdentity.dup());
  			DoubleMatrix zeros = DoubleMatrix.zeros(ARTICLE_FEAT_DIMEN);
  			vectorbt.put(id, zeros);
  			birthTimes.put(id, visitor.getTimestamp());
  			
  			//For Hybrid Model
  			matrixB.put(id, DoubleMatrix.zeros(ARTICLE_FEAT_DIMEN, K_CONST));
  		}
  		
  		//get articleFeature, using real ID;
  		ArticleFeatures aF = articleFeatureTable.get(article.getID());
  		for (int i = 0; i < ARTICLE_FEAT_DIMEN; i++) {
				double[] aFArr = aF.getFeatures();
				articleFeature.put(i, aFArr[i]);
			}
  		
  		DoubleMatrix zedVector = new DoubleMatrix(K_CONST);
  		for (int i = 0; i < K_CONST; i++) {
  			double val = articleFeature.get(i % ARTICLE_FEAT_DIMEN) * userFeature.get(i / USER_FEAT_DIMEN);
  			zedVector.put(i, val);
  		}
  		
  		// calculate the inverse by solving AX=I
  		DoubleMatrix matrixAtinversed = dInverse(matrixAt.get(id));
  		
  		
  		// calculate theta
  		DoubleMatrix tempforTheta=matrixB.get(id).dup().mmul(vectorBeta);
  		DoubleMatrix vectorTheta = matrixAtinversed.dup().mmul(vectorbt.get(id).dup().sub(
  													tempforTheta));
  		
  		//s_{t,a}
  		DoubleMatrix userFeatureTranspose = userFeature.transpose();
  		DoubleMatrix zedTranspose = zedVector.transpose();
  		
  		DoubleMatrix common24 = matrixA0inversed.dup().mmul(matrixB.get(id).transpose())
					.mmul(matrixAtinversed).mmul(userFeature);
			DoubleMatrix common34 = userFeatureTranspose.dup().mmul(matrixAtinversed);
			double s1 = zedTranspose.dot(matrixA0inversed.mmul(zedVector));
			double s2 = zedTranspose.dot(common24);
			double s3 = common34.dot(userFeature);
			double s4 = common34.dot(matrixB.get(id).mmul(common24));
  		
  		double s = s1-2*s2+s3+s4;
  		
  		//calculate P
  		double p = zedTranspose.dot(vectorBeta)
  					+ userFeatureTranspose.dot(vectorTheta)
  					+ ALPHA*Math.sqrt(s);
  		
  		// pick this article if p value is greater than max
  		if (p - max > THRESHOLD) {
  			max = p;
  			maxIndex = possibleActions.indexOf(article);
  			chosenID = article.getID();
  			chosenZed = zedVector;
  		}
  	}
  	
  	logLinesToUpdate--;
  	if (logLinesToUpdate == 0) {
	  	DoubleMatrix matrixBTranspose = matrixB.get(chosenID).transpose();
	  	DoubleMatrix matrixAtinversed = dInverse(matrixAt.get(chosenID));
	
	  	//////update shared part1////
	  	matrixA0 = matrixA0.add(matrixBTranspose.dup().mmul(matrixAtinversed).mmul(matrixB.get(chosenID)));
	  	vectorb0 = vectorb0.add(matrixBTranspose.dup().mmul(matrixAtinversed).mmul(vectorbt.get(chosenID)));
	  	
	  	//////update separate part////
	  	matrixAt.get(chosenID).add(userFeature.dup().mmul(userFeature.transpose()));
	  	matrixB.get(chosenID).add(userFeature.dup().mmul(chosenZed.transpose()));

	  	
	  	//////update shared part2////
	  	matrixAtinversed = dInverse(matrixAt.get(chosenID));
	  	matrixBTranspose = matrixB.get(chosenID).transpose();
	  	matrixA0 = matrixA0.add(chosenZed.dup().mmul(chosenZed.transpose()))
	  						.sub(matrixBTranspose.dup().mmul(matrixAtinversed).mmul(matrixB.get(chosenID)));
	  	vectorb0 = vectorb0.sub(matrixBTranspose.dup().mmul(matrixAtinversed).mmul(vectorbt.get(chosenID)));

	  	
	  	// reset interval
	  	logLinesToUpdate = UPDATE_INTERVAL;
  	}
  	
    return possibleActions.get(maxIndex);
  }

  private DoubleMatrix kInverse(DoubleMatrix matrix)
  {
	  return Solve.solve(matrix, kIdentity);
  }
  
  private DoubleMatrix dInverse(DoubleMatrix matrix) {
  	return Solve.solve(matrix, dIdentity);
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
			double[] chosenArticleFeature = articleFeatureTable.get(id).getFeatures();
			DoubleMatrix zed = new DoubleMatrix(K_CONST);
			for (int i = 0; i < K_CONST; i++) {
  			double val = chosenArticleFeature[i % ARTICLE_FEAT_DIMEN] * userFeature.get(i / USER_FEAT_DIMEN);
  			zed.put(i, val);
  		}
    	if (reward) {
    		vectorbt.get(id).add(userFeature);
    	}
    	
    	if(reward)
    	{
    		vectorb0 = vectorb0.add(zed);
    	}
  	}
  }

}

