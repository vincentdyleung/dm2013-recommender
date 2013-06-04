package myPolicy;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Hashtable;
import java.util.Scanner;

import org.ethz.las.bandit.logs.yahoo.ArticleFeatures;
import org.ethz.las.bandit.utils.ArrayHelper;

public class ArticleReader {

	private Scanner scan;
	
	public ArticleReader(String filePath) throws FileNotFoundException{
	    this.scan = new Scanner(new File(filePath));
	    this.scan.useDelimiter("\n");
	}
	
	public Hashtable<Integer, ArticleFeatures> read() throws IOException
	{
		Hashtable<Integer, ArticleFeatures> table = new Hashtable<Integer, ArticleFeatures>(MyPolicy.ARTICLE_COUNT);
		
		while (hasNext()) {
			ArticleFeatures features= readLine();
			table.put(features.getID(),features);
		}
		
		return table;
	}
	
	public ArticleFeatures readLine() throws IOException
	{
		if(! hasNext()) throw new IOException("no next line to read");
		
	    // Get next line with line scanner.
	    String line = scan.next();

	    // Tokenize the line.
	    String[] tokens = line.split("[\\s]+");

	    // Token 0 is the shown article ID
	    int articleId = Integer.parseInt(tokens[0]);
	    
	    // Tokens 1 - 6 are article features.
	    double [] features = ArrayHelper.stringArrayToDoubleArray(tokens, 1, 6);

	    //Article article = new Article(articleId);
	    ArticleFeatures articleFeatures = new ArticleFeatures(articleId, features);
	    
	    return articleFeatures;
	}
	
	  public boolean hasNext() throws IOException {
	    return scan.hasNext();
	  }
	  
	  public void close() throws IOException {
	  }
	  
}
