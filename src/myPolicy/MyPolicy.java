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
	
	// specified in the project description
	public static final int ARTICLE_COUNT = 271;
	public static final int ARTICLE_FEAT_DIMEN = 6;
	public static final int USER_FEAT_DIMEN = 6;
	public static final int K_CONST = USER_FEAT_DIMEN;//k
	
	// threshold to compare double values
	private static final double THRESHOLD = 0.00000000001;
	private DoubleMatrix ones;
	private DoubleMatrix identity;
	private DoubleMatrix[] matrixAt;
	private DoubleMatrix[] vectorbt;
	private DoubleMatrix userFeature;
	private DoubleMatrix articleFeature;
	private Random random;
	//private int chosenID;
	
	
	//For Hybrid Model
	private DoubleMatrix matrixA0;
	private DoubleMatrix vectorb0;
	private DoubleMatrix[] matrixB;
	private DoubleMatrix matrixA0inversed;
	private DoubleMatrix matrixAtinversed;
	
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
  	matrixA0 = DoubleMatrix.diag(ones, K_CONST, K_CONST);;
  	vectorb0 = DoubleMatrix.zeros(K_CONST);
  	matrixA0inversed = inversed(matrixA0);
  	
  	ArticleReader aReader = new ArticleReader(articleFilePath);
  	//read in article features
  	articleFeatureTable = aReader.read();
  	
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
		vectorBeta = matrixA0inversed.dup().mmul(vectorb0);
  	
  	// loop through availabe articles
  	for (Article article : possibleActions) {
  		// mod because article ID is not in the range of [0..270]
  		int id = article.getID() % ARTICLE_COUNT;

  		if (matrixAt[id] == null) {
  			// initialize if article is new
  			matrixAt[id] = DoubleMatrix.diag(ones, ARTICLE_FEAT_DIMEN, ARTICLE_FEAT_DIMEN);
  			vectorbt[id] = DoubleMatrix.zeros(ARTICLE_FEAT_DIMEN);
  			
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
  		matrixAtinversed = inversed(matrixAt[id]);
  		
  		// calculate theta
  		DoubleMatrix tempforTheta=matrixB[id].dup().mmul(vectorBeta);
  		DoubleMatrix vectorTheta = matrixAtinversed.dup().mmul(vectorbt[id].sub(
  													tempforTheta));
  		
  		//s_{t,a}
  		DoubleMatrix userFeatureTranspose = userFeature.transpose();
  		DoubleMatrix articleFeatureTranspose = articleFeature.transpose();
  		
  		DoubleMatrix common24 = matrixA0inversed.dup()
  													.mmul(matrixB[id].transpose())
														.mmul(matrixAtinversed);
  		double s1 = userFeatureTranspose.dup().mmul(matrixA0inversed).dot(userFeature);
  		double s2 = userFeatureTranspose.dup().mmul(common24).dot(articleFeature);
  		double s3 = articleFeatureTranspose.dup().mmul(matrixAtinversed).dot(articleFeature);
  		DoubleMatrix matrix4 = matrixAtinversed.dup().mmul(matrixB[id]).mmul(common24);
  		double s4 = articleFeatureTranspose.dup().mmul(matrix4).dot(articleFeature);
  		
  		double s = s1-2*s2+s3+s4;
  		
  		//calculate P
  		double p = articleFeatureTranspose.dot(vectorBeta)
  					+ userFeatureTranspose.dot(vectorTheta)
  					+ ALPHA*Math.sqrt(s);
  		
  		// break calculation of p into three steps
//  		DoubleMatrix intermediate = userFeature.transpose();intermediate.mmul(matrixAtinversed);
//  		double confidenceWidth = ALPHA * Math.sqrt(intermediate.dot(userFeature));
//  		double p = vectorTheta.transpose().dot(userFeature) + confidenceWidth;
  		
  		
  		// pick this article if p value is greater than max
  		if (p - max > THRESHOLD) {
  			max = p;
  			maxIndex = possibleActions.indexOf(article);
  			//chosenID = article.getID();
  		}
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
  	DoubleMatrix chosenArticleFeature = new DoubleMatrix(articleFeatureTable.get(a.getID()).getFeatures());
  	DoubleMatrix matrixBTranspose = matrixB[id].transpose();

  	//////update shared part1////
  	matrixA0 = matrixA0.addi(matrixBTranspose.dup().mmul(matrixAtinversed).mmul(matrixB[id]));
  	vectorb0 = vectorb0.addi(matrixBTranspose.dup().mmul(matrixAtinversed).mmul(vectorbt[id]));
  	
  	
  	//////update separate part////
  	matrixAt[id] = matrixAt[id].addi(chosenArticleFeature.dup().mmul(chosenArticleFeature.transpose()));
  	matrixB[id] = matrixB[id].addi(chosenArticleFeature.dup().mmul(userFeature.transpose()));
  	if (reward) {
  		vectorbt[id] = vectorbt[id].addi(chosenArticleFeature);
  	}
  	
  	//////update shared part2////
  	matrixBTranspose = matrixB[id].transpose();
  	matrixA0 = matrixA0.addi(userFeature.dup().mmul(userFeature.transpose()))
  						.subi(matrixBTranspose.dup().mmul(matrixAtinversed).mmul(matrixB[id]));
  	vectorb0 = vectorb0.subi(matrixBTranspose.dup().mmul(matrixAtinversed).mmul(vectorbt[id]));
  	if(reward)
  	{
  		vectorb0 = vectorb0.addi(userFeature);
  	}
  	
  }

}

